from openai import OpenAI
import os
import re
import json
import rdkit
import argparse
import pandas as pd
from tqdm import tqdm
from utils.dataset import MessageDataset, MessageDatasetWithExample


parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="quantized_models/llama3-70b/")
parser.add_argument("--port", type=int, default=8001)
# dataset settings
parser.add_argument("--data_folder", type=str, default="./data/ChEBI-20/raw/")
parser.add_argument("--example_folder", type=str, default="./data/ChEBI-20/raw/")
parser.add_argument("--task", type=str, default="molecule_points")
parser.add_argument("--mode", type=str, default="train")

parser.add_argument("--output_dir", type=str, default="./predictions_with_example/")

parser.add_argument("--temperature", type=float, default=0.75)
parser.add_argument("--top_p", type=float, default=0.85)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=512)

parser.add_argument("--seed", type=int, default=42)

# partition inference only works when batch_infer is False
parser.add_argument("--partition", type=int, default=1)
parser.add_argument("--cur", type=int, default=1)

parser.add_argument("--json_check", action="store_true", default=False)
parser.add_argument("--smiles_check", action="store_true", default=False)
parser.add_argument("--with_example", action="store_true", default=False)
parser.add_argument("--example_num", type=int, default=2)

args = parser.parse_args()

if 'bace' in args.data_folder or 'bbbp' in args.data_folder or 'clintox' in args.data_folder or 'hiv' in args.data_folder or 'muv' in args.data_folder or 'pcba' in args.data_folder or 'sider' in args.data_folder or 'tox21' in args.data_folder or 'toxcast' in args.data_folder:
    args.moleculenet = True
else:
    args.moleculenet = False

if "mistral" in args.base_model:
        args.mistral = True
else:
    args.mistral = False

# print parameters
print("========Parameters========")
for attr, value in args.__dict__.items():
    print("{}={}".format(attr.upper(), value))

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:{}/v1".format(args.port)

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# check out put dir
args.output_dir = args.output_dir + args.task + "-" + args.mode + "/"
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if os.path.exists(args.output_dir + "/output_" + args.task + "_" + str(args.cur) + ".csv"):
    temp = pd.read_csv(args.output_dir + "/output_" + args.task + "_" + str(args.cur) + ".csv")
    start_pos = len(temp)
else:
    with open(args.output_dir + "/output_" + args.task + "_" + str(args.cur) + ".csv", "w+") as f:
        f.write("outputs\n")
    start_pos = 0
print("========Inference Init========")
print("Start from: ", start_pos)
print("==============================")

# load dataset
inference_dataset = MessageDataset(args.data_folder, args.task, args.mode) if not args.with_example else MessageDatasetWithExample(args.data_folder, args.task, args.mode, args.example_folder, args.example_num, args.moleculenet, args.mistral)
print(inference_dataset[0])
print(len(inference_dataset))

error_records = []

with tqdm(total=int(len(inference_dataset)*args.cur/args.partition)-int(len(inference_dataset)*(args.cur-1)/args.partition)-start_pos) as pbar:
    for idx in range(int(len(inference_dataset)*(args.cur-1)/args.partition)+start_pos, int(len(inference_dataset)*args.cur/args.partition)):
        error_allowance = 0
        while True:
            completion = client.chat.completions.create(
                model=args.base_model,
                messages=inference_dataset[idx],
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                n=args.num_return_sequences,
                stop=["</s>", "<|end_of_text|>", "<|eot_id|>"],
                seed=args.seed
            )
            s = completion.choices[0].message.content

            if args.json_check:
                match = re.search(r'\{.*?\}', s, re.DOTALL)
                if match:

                    if args.smiles_check:
                        json_str = match.group()
                        try:
                            json_obj = json.loads(json_str)
                            molecule = json_obj["molecule"]
                            if rdkit.Chem.MolFromSmiles(molecule):
                                break
                            else:
                                # change random seed
                                args.seed += 1
                                error_allowance += 1
                                if error_allowance > 10:
                                    error_records.append(idx)
                                    break
                        except:
                            # change random seed
                            args.seed += 1
                            error_allowance += 1
                            if error_allowance > 10:
                                error_records.append(idx)
                                break
                    else:
                        break
                else:
                    # change random seed
                    args.seed += 1
                    error_allowance += 1
                    if error_allowance > 10:
                        error_records.append(idx)
                        break
            else:
                break
        print(s)

        df = pd.DataFrame([s.strip()], columns=["outputs"])
        df.to_csv(args.output_dir + "/output_" + args.task + "_" + str(args.cur) + ".csv", mode='a', header=False, index=False)
        # with open(args.output_dir + "/output_" + args.task + ".txt", "a+") as f:
        #     f.write(s.replace('\n', ' ').strip() + "\n")
        pbar.update(1)


print("========Inference Done========")
print("Error Records: ", error_records)