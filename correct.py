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
parser.add_argument("--task", type=str, default="MolEdit/")
parser.add_argument("--subtask", type=str, default="AddComponent")
args = parser.parse_args()

args.input = "./{}/{}/open_generation/{}/{}.csv".format(args.folder, args.name, args.task, args.subtask)

args.output = args.input.replace(".csv", "_corrected.csv")

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