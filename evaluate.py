'''
For evaluation
'''
import argparse
import pandas as pd
from utils.evaluation import mol_prop, calculate_novelty, calculate_similarity
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="llama3-70B")

# dataset settings
parser.add_argument("--benchmark", type=str, default="open_generation")
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="AtomNum")

parser.add_argument("--output_dir", type=str, default="./predictions/")
parser.add_argument("--calc_novelty", action="store_true", default=False)

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
            valid_molecules = []
            
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
                if mol_prop(target["outputs"][idx], "validity"):
                    valid_molecules.append(target["outputs"][idx])
            
            
            print("Accuracy: ", sum(flags) / len(flags))
            print("Validty:", len(valid_molecules) / len(flags))
            if args.calc_novelty:
                novelties = calculate_novelty(valid_molecules)
                print("Novelty: ", sum(novelties) / len(novelties))
                
        elif args.subtask == "BasicProp":
            # TODO: Implement this
            pass
        elif args.subtask == "FunctionalGroup":
            functional_groups = ['benzene rings', 'hydroxyl', 'anhydride', 'aldehyde', 'ketone', 'carboxyl', 'ester', 'amide', 'amine', 'nitro', 'halo', 'nitrile', 'thiol', 'sulfide', 'disulfide', 'sulfoxide', 'sulfone', 'phosphate', 'borane', 'borate', 'borohydride']
            flags = []
            valid_molecules = []
            for idx in tqdm(range(len(data))):
                flag = 1
                for group in functional_groups:
                    if group == "benzene rings":
                        if mol_prop(target["outputs"][idx], "num_benzene_ring") != int(data[group][idx]):
                            flag = 0
                            break
                    else:
                        if mol_prop(target["outputs"][idx], "num_" + group) != int(data[group][idx]):
                            flag = 0
                            break
                flags.append(flag)
                if mol_prop(target["outputs"][idx], "validity"):
                    valid_molecules.append(target["outputs"][idx])
                
            print("Accuracy: ", sum(flags) / len(flags))
            print("Validty:", len(valid_molecules) / len(flags))
            if args.calc_novelty:
                novelties = calculate_novelty(valid_molecules)
                print("Novelty: ", sum(novelties) / len(novelties))

    elif args.task == "MolEdit":
        if args.subtask == "AddComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(data))):
                raw = data["molecule"][idx]
                group = data["added_group"][idx]
                if group == "benzene ring":
                    group = "benzene_ring"
                target_mol = target["outputs"][idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)

                    if mol_prop(target_mol, "num_" + group) == mol_prop(raw, "num_" + group) + 1:
                        successed.append(1)
                    else:
                        successed.append(0)

                    similarities.append(calculate_similarity(raw, target_mol))
                else:
                    successed.append(0)

            print("Success Rate:", sum(successed) / len(successed))
            print("Validty:", len(valid_molecules) / len(data))
            print("Similarity:", sum(similarities) / len(similarities))

        elif args.subtask == "DelComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(data))):
                raw = data["molecule"][idx]
                group = data["removed_group"][idx]
                if group == "benzene ring":
                    group = "benzene_ring"
                target_mol = target["outputs"][idx]
                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)

                    if mol_prop(target_mol, "num_" + group) == mol_prop(raw, "num_" + group) - 1:
                        successed.append(1)
                    else:
                        successed.append(0)

                    similarities.append(calculate_similarity(raw, target_mol))
                else:
                    successed.append(0)

            print("Success Rate:", sum(successed) / len(successed))
            print("Validty:", len(valid_molecules) / len(data))
            print("Similarity:", sum(similarities) / len(similarities))
        elif args.subtask == "SubComponent":
            valid_molecules = []
            successed = []
            similarities = []
            for idx in tqdm(range(len(data))):
                raw = data["molecule"][idx]
                added_group = data["added_group"][idx]
                removed_group = data["removed_group"][idx]
                if added_group == "benzene ring":
                    added_group = "benzene_ring"
                if removed_group == "benzene ring":
                    removed_group = "benzene_ring"

                target_mol = target["outputs"][idx]

                if mol_prop(target_mol, "validity"):
                    valid_molecules.append(target_mol)

                    if mol_prop(target_mol, "num_" + removed_group) == mol_prop(raw, "num_" + removed_group) - 1 and mol_prop(target_mol, "num_" + added_group) == mol_prop(raw, "num_" + added_group) + 1:
                        successed.append(1)
                    else:
                        successed.append(0)

                    similarities.append(calculate_similarity(raw, target_mol))
                else:
                    successed.append(0)

            print("Success Rate:", sum(successed) / len(successed))
            print("Validty:", len(valid_molecules) / len(data))
            print("Similarity:", sum(similarities) / len(similarities))

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