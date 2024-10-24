'''
Almost all the unmatches come from the "" "" in the text. 
'''
import os
import re
import json
import pandas as pd
from argparse import ArgumentParser


def correct_text(text):
    # first find json
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            json_str = json_str.replace('""', '"')

            print(json_str)
            try:
                json_obj = json.loads(json_str)
                s = json_obj["molecule"]
                if "=>" in s:
                    s = s.split("=>")[-1].strip()
                if "->" in s:
                    s = s.split("->")[-1].strip()
                print(s)
                return s
            except Exception as e:
                s = json_str.split(":")[1].strip().strip('}').strip().strip('"').strip()
                if "=>" in s:
                    s = s.split("=>")[-1].strip()
                if "->" in s:
                    s = s.split("->")[-1].strip()
                print(s)
                return s
        else:
            s = text.replace('\n', ' ').strip()
            if "=>" in s:
                    s = s.split("=>")[-1].strip()
            if "->" in s:
                s = s.split("->")[-1].strip()
            return s
    except Exception as e:
        print(e, text)
        return "None"


parser = ArgumentParser()
parser.add_argument("--folder", type=str, default="new_predictions")
parser.add_argument("--name", type=str, default="llama3-8B")
parser.add_argument("--task", type=str, default="MolOpt/")
parser.add_argument("--subtask", type=str, default="LogP")
args = parser.parse_args()

args.input = "./{}/{}/open_generation/{}/{}.csv".format(args.folder, args.name, args.task, args.subtask)

args.output = args.input.replace(".csv", "_corrected.csv")

## First check line number == 5001?
with open(args.input, "r") as f:
    lines = f.readlines()
temp = []
if len(lines) != 5001:
    for line in lines:
        if line[0] == "0":
            temp.append(line)
    # check again
    if len(temp) != 5001:
        print("The file does not have 5001 lines. Please check the file.")
        exit(0)
    else:
        args.input = args.input.replace(".csv", "_temp.csv")
        with open(args.input, "w+") as f:
            f.writelines(temp)
        

data = pd.read_csv(args.input)

new_data = []
for idx, row in data.iterrows():
    new_data.append(correct_text(row["outputs"]))

    
df = pd.DataFrame(new_data, columns=["outputs"])
df.to_csv(args.output, index=False)

confirmation = input("Do you want to overwrite the original file? (y/n): ")
if confirmation == "y":
    # backup the original file
    backup_file = args.input.replace(".csv", "_backup.csv")
    os.system("mv {} {}".format(args.input, backup_file))
    os.system("mv {} {}".format(args.output, args.input))

else:
    print("The corrected file is saved at:", args.output)