'''
For evaluation
'''
import argparse
import pandas as pd
from utils.evaluation import mol_prop
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="quantized_models/llama3-70b/")
parser.add_argument("--name", type=str, default="llama3-70B")

# dataset settings
parser.add_argument("--benchmark", type=str, default="open_generation")
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="AtomNum")

parser.add_argument("--output_dir", type=str, default="./predictions/")


args = parser.parse_args()

raw_file = "./data/benchmarks/{}/{}/{}/test.csv".format(args.benchmark, args.task, args.subtask)
target_file = args.output_dir + args.name + "/" + args.benchmark + "/" + args.task + "/" + args.subtask + ".csv"

data = pd.read_csv(raw_file)
target = pd.read_csv(target_file)

if args.benchmark == "open_generation":
    if args.task == "MolCustom":
        if args.subtask == "AtomNum":
            # accuracy
            atom_type = ['carbon', 'oxygen', 'nitrogen', 'sulfur', 'fluorine', 'chlorine', 'bromine', 'iodine', 'phosphorus', 'boron', 'silicon', 'selenium', 'tellurium', 'arsenic', 'antimony', 'bismuth', 'polonium']
            flags = []
            novelties = []
            validities = []
            # use tqdm to show the progress
            for idx in tqdm(range(len(data))):
                flag = 1
                for atom in atom_type:
                    if mol_prop(target["outputs"][idx], "num_" + atom) != int(data[atom][idx]):
                        flag = 0
                        break
                flags.append(flag)
                # Novelty
                # novelty = mol_prop(target["outputs"][idx], "novelty")
                # if novelty is not None:
                #     novelties.append(novelty)
                validities.append(1 if mol_prop(target["outputs"][idx], "validity") else 0)
                
            print("Accuracy: ", sum(flags) / len(flags))
            # print("Novelty: ", sum(novelties) / len(novelties))
            print("Validty:", sum(validities) / len(validities))
                
        elif args.subtask == "BasicProp":
            pass
        elif args.subtask == "FunctionalGroup":
            pass

    elif args.task == "MolEdit":
        if args.subtask == "AddComponent":
            pass
        elif args.subtask == "DelComponent":
            pass
        elif args.subtask == "SubComponent":
            pass

    elif args.task == "MolOpt":
        if args.subtask == "LogP":
            pass
        elif args.subtask == "MR":
            pass
        elif args.subtask == "QED":
            pass
elif args.benchmark == "targeted_generation":
    pass
else:
    raise ValueError("Invalid Benchmark Type")