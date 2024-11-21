# -*- coding: utf8 -*-
import requests
import json
import os
import re
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
parser.add_argument("--name", type=str, default="GPT-4o")

parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--port", type=int, default=8000)
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

# print parameters
print("========Parameters========")
for attr, value in args.__dict__.items():
    print("{}={}".format(attr.upper(), value))

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

Baseurl = "https://api.claudeshop.top"
if "gpt" in args.model:
    Skey = "sk-sY1NKbcyzHaLIbQsoNyafYhx4OK8UWo4vbLmQTSWItiTOWHc"
else:
    Skey = "sk-JeUHXQzwTsm5y7wVY5DAgnklttY9SuRdIQf8HfhEbIsW3Fjj"

"""
payload = json.dumps({
   "model": "gpt-4o",
   "messages": [
      {
         "role": "system",
         "content": "You are a helpful assistant."
      },
      {
         "role": "user",
         "content": "hello"
      }
   ]
})
"""

url = Baseurl + "/v1/chat/completions"
headers = {
   'Accept': 'application/json',
   'Authorization': f'Bearer {Skey}',
   'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
   'Content-Type': 'application/json'
}

"""
response = requests.request("POST", url, headers=headers, data=payload)

# 解析 JSON 数据为 Python 字典
data = response.json()

# 获取 content 字段的值
content = data

print(content["choices"][0]["message"]["content"])
"""

error_records = []

with tqdm(total=len(inference_dataset)-start_pos) as pbar:
    for idx in range(start_pos, len(inference_dataset)):
        cur_seed = args.seed
        error_allowance = 0
        while True:
            try:
                
                prompt = inference_dataset[idx]
                prompt[0]["content"] = prompt[0]["content"] + " Remember you should only return the JSON object without answering anything else."
                payload = json.dumps({
                            "model": args.model,
                            "messages": prompt
                    })
                response = requests.request("POST", url, headers=headers, data=payload)

                data = response.json()
                content = data
                s = content["choices"][0]["message"]["content"]
            
            
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

