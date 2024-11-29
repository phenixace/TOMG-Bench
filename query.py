'''
For vllm query inference
'''
from openai import OpenAI
import os
import re
import json
import rdkit
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from utils.dataset import OMGDataset, TMGDataset
import transformers
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="quantized_models/llama3-70b/")
parser.add_argument("--name", type=str, default="llama3-70B")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--load_lora", type=bool, default=False)
parser.add_argument("--lora_model_path", type=str, default="")
# dataset settings
parser.add_argument("--benchmark", type=str, default="open_generation")
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="AtomNum")


parser.add_argument("--output_dir", type=str, default="./predictions/")

parser.add_argument("--temperature", type=float, default=0.75)
parser.add_argument("--top_p", type=float, default=0.85)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=512)

parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--json_check", action="store_true", default=False)
parser.add_argument("--smiles_check", action="store_true", default=False)

# add a log option to record the output
parser.add_argument("--log", action="store_true", default=False)

args = parser.parse_args()

if "mistral" in args.model:
        args.mistral = True
else:
    args.mistral = False

# print parameters
print("========Parameters========")
for attr, value in args.__dict__.items():
    print("{}={}".format(attr.upper(), value))

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://10.140.24.31:{}/v1".format(args.port)

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


# check out put dir
args.output_dir = args.output_dir + args.name + "/" + args.benchmark + "/" + args.task + "/"
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if os.path.exists(args.output_dir + args.subtask + ".csv"):
    temp = pd.read_csv(args.output_dir + args.subtask + ".csv")
    start_pos = len(temp)
else:
    with open(args.output_dir + args.subtask + ".csv", "w+") as f:
        f.write("outputs\n")
    start_pos = 0

print("========Inference Init========")
print("Inference starts from: ", start_pos)


# load dataset
if args.benchmark == "open_generation":
    inference_dataset = OMGDataset(args.task, args.subtask, args.json_check)
elif args.benchmark == "targeted_generation":
    inference_dataset = TMGDataset(args.task, args.subtask, args.json_check)
print("========Sanity Check========")
print(inference_dataset[0])
print("Total length of the dataset:", len(inference_dataset))
print("==============================")

error_records = []

if args.load_lora == True:
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True, **kwargs)
    print(f"Loading LoRA weights from {args.lora_model_path}")
    model = PeftModel.from_pretrained(model, args.lora_model_path)
    print(f"Merging weights")
    model = model.merge_and_unload()
    print('Convert to BF16...')
    model = model.to(torch.bfloat16)

else:
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()


pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer, 
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        temperature=args.temperature,
        trust_remote_code=True,
        top_p=args.top_p,
)


with tqdm(total=len(inference_dataset)-start_pos) as pbar:
    for idx in range(start_pos, len(inference_dataset)):
        cur_seed = args.seed
        error_allowance = 0
        while True:
            try:
                """
                    completion = client.chat.completions.create(
                        model=args.model,
                        messages=inference_dataset[idx],
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        n=args.num_return_sequences,
                        stop=["</s>", "<|end_of_text|>", "<|eot_id|>"],
                        seed=cur_seed
                    )
                    s = completion.choices[0].message.content
                """
                    
                prompt = inference_dataset[idx]
                inputs = tokenizer.apply_chat_template(prompt,
                                            add_generation_prompt=True,
                                            tokenize=True,
                                            return_tensors="pt",
                                            return_dict=True
                                            )
                inputs = inputs.to(device)
                gen_kwargs = {"max_length": args.max_new_tokens, "do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
                #outputs = pipeline(prompt, max_new_tokens=args.max_new_tokens)
                with torch.no_grad():
                        outputs = model.generate(**inputs, **gen_kwargs)
                        outputs = outputs[:, inputs['input_ids'].shape[1]:]
                        s = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            
            except:
                # change random seed
                cur_seed += 1
                error_allowance += 1
                if error_allowance > 10:
                    s = "None"   # empty string
                    error_records.append(idx)
                    break
                else:
                    continue
            
            
            s = s.replace('""', '"').strip()
            print("Raw:", s)

            if s == None:
                cur_seed += 1
                error_allowance += 1
                if error_allowance > 10:
                    s = ""   # empty string
                    error_records.append(idx)
                    break
                else:
                    continue

            if args.log:
                with open(args.output_dir + args.subtask + ".log", "a+") as f:
                    f.write(s.replace('\n', ' ').strip() + "\n")

            if args.json_check:
                match = re.search(r'\{.*?\}', s, re.DOTALL)
                if match:
                    json_str = match.group()
                    try:
                        json_obj = json.loads(json_str)
                        s = json_obj["molecule"]
                        # add smiles check
                        if args.smiles_check:
                            try:
                                mol = Chem.MolFromSmiles(s)
                                if mol is None:
                                    cur_seed += 1
                                    error_allowance += 1
                                    if error_allowance > 10:
                                        error_records.append(idx)
                                        break
                                    else:
                                        continue
                            except:
                                cur_seed += 1
                                error_allowance += 1
                                if error_allowance > 10:
                                    error_records.append(idx)
                                    break
                                else:
                                    continue
                        break
                    except:
                        # change random seed
                        cur_seed += 1
                        error_allowance += 1
                        if error_allowance > 10:
                            error_records.append(idx)
                            break
                        else:
                            continue

                else:
                    # change random seed
                    cur_seed += 1
                    error_allowance += 1
                    if error_allowance > 10:
                        error_records.append(idx)
                        break
                    else:
                        continue
            else:
                break
        print("Checked:", s)
        
        # check again
        if not isinstance(s, str):
            s = str(s)

        s = s.replace('\n', ' ').strip()  # remove newline characters

        df = pd.DataFrame([s.strip()], columns=["outputs"])
        df.to_csv(args.output_dir +  args.subtask + ".csv", mode='a', header=False, index=True)
        # with open(args.output_dir + "/output_" + args.task + ".txt", "a+") as f:
        #     f.write(s.replace('\n', ' ').strip() + "\n")
        pbar.update(1)


print("========Inference Done========")
print("Error Records: ", error_records)
