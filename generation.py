from argparse import ArgumentParser
import random
import pandas as pd
import copy
from utils.evaluation import mol_prop


parser = ArgumentParser()
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="BasicProp")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

def sample_with_weights(population, raw_weights, sample_size):
    weights = copy.deepcopy(raw_weights)
    sample = []
    while len(sample) < sample_size:
        chosen = random.choices(population, weights=weights, k=1)[0]
        if chosen not in sample:
            sample.append(chosen)
            # set the weight of the chosen item to 0
            weights[population.index(chosen)] = 0
    return sample

file_dir = f'./data/benchmarks/open_generation/{args.task}/{args.subtask}/'

if args.task == "MolEdit":
    if args.subtask == "AddComponent":
        FunctionalGroups = ["benzene ring", "hydroxyl", "aldehyde", "carboxyl", "amide", "amine", "nitro", "halo", "nitrile", "thiol"]
        groups_weights = [15, 15, 5, 5, 10, 5, 5, 5, 1, 1]

        prompt_templates = ["Please add a {} to the molecule {}.", "Modify the molecule {} by adding a {}.", "Add a {} to the molecule {}."]

        Instructions = {"index":[], "Instruction":[], "molecule":[], "added_group":[]}

        data = pd.read_csv(file_dir + '/test_raw.csv')
        for i in range(len(data)):
            molecule = data.iloc[i]['smiles']
            index = data.iloc[i]['index']

            molecule = molecule.strip().strip('\n').strip()

            # randomly select a functional group to add
            to_add = random.choices(FunctionalGroups, groups_weights, k=1)[0]

            temp_num = random.randint(0, 2)
            if temp_num == 1:
                text = prompt_templates[temp_num].format(molecule, to_add)
            else:
                text = prompt_templates[temp_num].format(to_add, molecule)

            Instructions["index"].append(index)
            Instructions["Instruction"].append(text)
            Instructions["molecule"].append(molecule)
            Instructions["added_group"].append(to_add)

        df = pd.DataFrame(Instructions)
        df.to_csv(file_dir + "/test.csv", index=False)
    elif args.subtask == "DelComponent":
        # examine functional groups in the molecule
        FunctionalGroups = ["benzene_ring", "hydroxyl", "aldehyde", "carboxyl", "amide", "amine", "nitro", "halo", "nitrile", "thiol"]

        prompt_templates = ["Please remove a {} from the molecule {}.", "Modify the molecule {} by removing a {}.", "Remove a {} from the molecule {}."]
        data = pd.read_csv(file_dir + '/test_raw.csv')

        Instructions = {"index":[], "Instruction":[], "molecule":[], "removed_group":[]}
        number = 0
        for i in range(len(data)):
            molecule = data.iloc[i]['smiles']
            index = data.iloc[i]['index']

            molecule = molecule.strip().strip('\n').strip()
            to_removes = []
            for group in FunctionalGroups:
                if mol_prop(molecule, "num_"+group) > 0:
                    to_removes.append(group)
                    
            if len(to_removes) == 0:
                continue
            else:
                to_remove = random.choice(to_removes)
            
            temp_num = random.randint(0, 2)
            if temp_num == 1:
                text = prompt_templates[temp_num].format(molecule, to_remove)
            else:
                text = prompt_templates[temp_num].format(to_remove, molecule)

            Instructions["index"].append(index)
            Instructions["Instruction"].append(text)
            Instructions["molecule"].append(molecule)
            Instructions["removed_group"].append(to_remove)
            number += 1
            if number == 5000:
                break

        df = pd.DataFrame(Instructions)
        df.to_csv(file_dir + "/test.csv", index=False)
    elif args.subtask == "SubComponent":
        # examine functional groups in the molecule
        FunctionalGroups = ["hydroxyl", "aldehyde", "carboxyl", "amide", "amine", "nitro", "halo", "nitrile", "thiol"]

        prompt_templates = ["Please substitute a {} in the molecule {} by {}.", "Modify the molecule {} by replacing a {} by {}.", "Replace a {} in the molecule {} by {}.", "Please replace a {} in the molecule {} with {}.", "Modify the molecule {} by substituting a {} with {}.", "Substitute a {} in the molecule {} with {}."]
        data = pd.read_csv(file_dir + '/test_raw.csv')

        Instructions = {"index":[], "Instruction":[], "molecule":[], "removed_group":[], "added_group":[]}
        number = 0
        for i in range(len(data)):
            molecule = data.iloc[i]['smiles']
            index = data.iloc[i]['index']

            molecule = molecule.strip().strip('\n').strip()
            to_removes = []
            for group in FunctionalGroups:
                if mol_prop(molecule, "num_"+group) > 0:
                    to_removes.append(group)
                    
            if len(to_removes) == 0:
                continue
            else:
                to_remove = random.choice(to_removes)

            temp_toadds = [item for item in FunctionalGroups if item != to_remove]
            to_add = random.choice(temp_toadds)

            temp_num = random.randint(0, 5)
            if temp_num == 1 or temp_num == 4:
                text = prompt_templates[temp_num].format(molecule, to_remove, to_add)
            else:
                text = prompt_templates[temp_num].format(to_remove, molecule, to_add)
            Instructions["index"].append(index)
            Instructions["Instruction"].append(text)
            Instructions["molecule"].append(molecule)
            Instructions["removed_group"].append(to_remove)
            Instructions["added_group"].append(to_add)
            number += 1
            if number == 5000:
                break

        df = pd.DataFrame(Instructions)
        df.to_csv(file_dir + "/test.csv", index=False)
elif args.task == 'MolCustom':
    if args.subtask == 'AtomNum':  # dataset needs to be update
        elements = ["oxygen", "nitrogen", "sulfur", "fluorine", "chlorine", "bromine", "iodine", "phosphorus", "boron", "silicon", "selenium", "tellurium", "arsenic", "antimony", "bismuth", "polonium"]
        elements_weights = [5, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        elements_num = [1, 2, 3, 4, 5]
        elements_num_weights = [10, 5, 2, 1, 1]

        prompt_templates = ["Please generate a molecule with ", "Please generate a molecule composed of ", "Please generate a molecule consisting ", "The molecule has ", "The molecule is composed of ", "The molecule consists of ", "There is a molecule with ", "There is a molecule composed of ", "There is a molecule consisting of ", "The molecule contains "]

        instructions = {"Instruction":[], "carbon":[], "oxygen":[], "nitrogen":[], "sulfur":[], "fluorine":[], "chlorine":[], "bromine":[], "iodine":[], "phosphorus":[], "boron":[], "silicon":[], "selenium":[], "tellurium":[], "arsenic":[], "antimony":[], "bismuth":[], "polonium":[]}
        i = 0
        while i < 5000:
            
            carbon_num = random.randint(1, 40)
            candidate = random.choice(prompt_templates) + str(carbon_num) + " carbon atoms"

            other_elements_num = [item for item in range(0, min(5, carbon_num))]
            other_elements_num_weights = [int(10/(item+1)) for item in other_elements_num]
            other_elements = random.choices(other_elements_num, other_elements_num_weights, k=1)[0]

            Other_elements_dict = {"oxygen":0, "nitrogen":0, "sulfur":0, "fluorine":0, "chlorine":0, "bromine":0, "iodine":0, "phosphorus":0, "boron":0, "silicon":0, "selenium":0, "tellurium":0, "arsenic":0, "antimony":0, "bismuth":0, "polonium":0}
            if other_elements == 0:
                candidate += "."
            elif other_elements == 1:
                element = random.choices(elements, elements_weights, k=1)[0]
                element_num = random.choices(elements_num, elements_num_weights, k=1)[0]
                candidate += " and " + str(element_num) + " " + element + " atom."
                Other_elements_dict[element] = element_num
            else:
                candidate += ", "
                temp_elements = sample_with_weights(elements, elements_weights, other_elements)
                
                temp_elements_num = sample_with_weights(elements_num, elements_num_weights, other_elements)
                for j in range(len(temp_elements)):
                    if j == other_elements - 1:
                        candidate += "and " + str(temp_elements_num[j]) + " " + temp_elements[j] + " atoms."
                    else:
                        candidate += str(temp_elements_num[j]) + " " + temp_elements[j] + " atoms, "

                    Other_elements_dict[temp_elements[j]] = temp_elements_num[j]
                
            if candidate not in instructions["Instruction"]:
                instructions["Instruction"].append(candidate)
                instructions["carbon"].append(carbon_num)
                for key in Other_elements_dict.keys():
                    instructions[key].append(Other_elements_dict[key])
                i += 1
        df = pd.DataFrame(instructions)
        df.to_csv(file_dir + "/test.csv", index=False)
    elif args.subtask == 'BasicProp':
        prompt_templates = []
        prompt_templates.append(["Please generate a molecule with {} property.", "The molecule has {} property.", "There is a molecule with {} property.", "Please generate a {} molecule.", "The molecule is a {} molecule.", "There is a {} molecule.", "The molecule is {}."])
        prompt_templates.append(["Please generate a molecule with {} and {} properties.", "The molecule has {} and {} properties.", "There is a molecule with {} and {} properties.", "Please generate a molecule with {} and {} properties.", "The molecule is a {} and {} molecule.", "There is a {} and {} molecule.", "The molecule is {} and {}."])
        prompt_templates.append(["Please generate a molecule with {}, {} and {} properties.", "The molecule has {}, {} and {} properties.", "There is a molecule with {}, {} and {} properties.", "Please generate a molecule with {}, {} and {} properties.", "The molecule is a {}, {} and {} molecule.", "There is a {}, {} and {} molecule.", "The molecule is {}, {} and {}."])

        prop_noun = ["toxicity", "non-toxicity", "high boling point", "low boiling point", "high melting point", "low melting point", "soluble in water", "insoluble in water"]
        prop_adj = ["toxic", "non-toxic", "high-boiling", "low-boiling", "high-melting", "low-melting", "water-soluble", "water-insoluble", "heavy", "light", "complex", "simple"]

        Instructions = {"Instruction":[], "property":[]}
        for i in range(0,5000):
            num_prop = random.randint(1, 3)
            temp_template = random.choice(prompt_templates[num_prop-1])

            adj_or_noun = random.randint(0, 1)

            props = []
            if adj_or_noun == 0:
                box = [0,1,2,3]
                box_weights = [1, 1, 1, 1]
                cur_prop = sample_with_weights(box, box_weights, num_prop)
                for j in range(num_prop):
                    pos_neg = random.randint(0, 1)
                    props.append(prop_noun[2*cur_prop[j] + pos_neg])
            else:
                box = [0,1,2,3,4,5]
                box_weights = [1, 1, 1, 1, 1, 1]
                cur_prop = sample_with_weights(box, box_weights, num_prop)
                for j in range(num_prop):
                    pos_neg = random.randint(0, 1)
                    props.append(prop_adj[2*cur_prop[j] + pos_neg])
            if num_prop == 1:
                text = temp_template.format(props[0])
            elif num_prop == 2:
                text = temp_template.format(props[0], props[1])
            else:
                text = temp_template.format(props[0], props[1], props[2])
            Instructions["Instruction"].append(text)
            if num_prop == 1:
                Instructions["property"].append(props[0])
            else:
                Instructions["property"].append(",".join(props))

        df = pd.DataFrame(Instructions)
        df.to_csv(file_dir + "/test.csv", index=False)
    elif args.subtask == 'FunctionalGroup':
        groups = ["benzene rings", "hydroxyl", "anhydride", "aldehyde", "ketone", "carboxyl", "ester", "amide", "amine", "nitro", "halo", "thioether", "nitrile", "thiol", "sulfide", "disulfide", "sulfoxide", "sulfone", "phosphate", "borane", "borate", "borohydride"]
        groups_weights = [15, 15, 2, 5, 5, 10, 5, 5, 5, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        groups_num = [1, 2, 3, 4, 5]
        groups_num_weights = [10, 5, 2, 1, 1]


        prompt_templates = ["Please generate a molecule with ", "Please generate a molecule composed of ", "Please generate a molecule consisting ", "The molecule has ", "The molecule is composed of ", "The molecule consists of ", "There is a molecule with ", "There is a molecule composed of ", "There is a molecule consisting of ", "The molecule contains "]

        instructions = {"Instruction":[], "benzene rings":[], "hydroxyl":[], "anhydride":[], "aldehyde":[], "ketone":[], "carboxyl":[], "ester":[], "amide":[], "amine":[], "nitro":[], "halo":[], "thioether":[], "nitrile":[], "thiol":[], "sulfide":[], "disulfide":[], "sulfoxide":[], "sulfone":[], "phosphate":[], "borane":[], "borate":[], "borohydride":[]}
        i = 0
        while i < 5000:
            candidate = random.choice(prompt_templates)

            other_groups_num = [item for item in range(1, 5)]
            other_groups_num_weights = [int(10/(item)) for item in other_groups_num]
            other_groups = random.choices(other_groups_num, other_groups_num_weights, k=1)[0]

            temp_groups_dict = {"benzene rings":0, "hydroxyl":0, "anhydride":0, "aldehyde":0, "ketone":0, "carboxyl":0, "ester":0, "amide":0, "amine":0, "nitro":0, "halo":0, "thioether":0, "nitrile":0, "thiol":0, "sulfide":0, "disulfide":0, "sulfoxide":0, "sulfone":0, "phosphate":0, "borane":0, "borate":0, "borohydride":0}

            if other_groups == 1:
                group = random.choices(groups, groups_weights, k=1)[0]
                group_num = random.choices(groups_num, groups_num_weights, k=1)[0]
                temp_groups_dict[group] = group_num
                candidate += str(group_num) + " " + group + " groups."
            else:
                temp_groups = sample_with_weights(groups, groups_weights, other_groups)
                temp_groups_num = sample_with_weights(groups_num, groups_num_weights, other_groups)
                for j in range(len(temp_groups)):
                    if j == other_groups - 1:
                        candidate += "and " + str(temp_groups_num[j]) + " " + temp_groups[j] + " groups."
                    else:
                        candidate += str(temp_groups_num[j]) + " " + temp_groups[j] + " groups, "
                    temp_groups_dict[temp_groups[j]] = temp_groups_num[j]

            if candidate not in instructions["Instruction"]:
                instructions["Instruction"].append(candidate)
                for key in temp_groups_dict.keys():
                    instructions[key].append(temp_groups_dict[key])
                i += 1
        df = pd.DataFrame(instructions)
        df.to_csv(file_dir + "/test.csv", index=False)
    elif args.subtask == "BondNum":
        bonds_type = ["single", "double", "triple", "rotatable", "aromatic"]
        bonds_type_weights = [5, 4, 3, 1, 1]
        
        bonds_num = [1, 2, 3, 4, 5]
        bonds_num_weights = [10, 5, 2, 1, 1]


        prompt_templates = ["Please generate a molecule with ", "Please generate a molecule composed of ", "Please generate a molecule consisting ", "The molecule has ", "The molecule is composed of ", "The molecule consists of ", "There is a molecule with ", "There is a molecule composed of ", "There is a molecule consisting of ", "The molecule contains "]

        instructions = {"Instruction":[], "single":[], "double":[], "triple":[], "rotatable":[], "aromatic":[]}
        i = 0
        while i < 5000:
            candidate = random.choice(prompt_templates)

            other_bonds_num = [item for item in range(1, 5)]
            other_bonds_num_weights = [int(10/(item)) for item in other_bonds_num]
            other_bonds = random.choices(other_bonds_num, other_bonds_num_weights, k=1)[0]

            temp_bonds_dict = {"single":0, "double":0, "triple":0, "rotatable":0, "aromatic":0}

            if other_bonds == 1:
                bond = random.choices(bonds_type, bonds_type_weights, k=1)[0]
                if bond == "aromatic":
                    bond_num = random.choices([5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [5, 20, 5, 5, 5, 1, 5, 1, 5, 1, 1, 1, 1, 1, 1], k=1)[0]
                elif bond == "single":
                    bond_num = random.randint(1, 50)
                else:
                    bond_num = random.choices(bonds_num, bonds_num_weights, k=1)[0]

                temp_bonds_dict[bond] = bond_num
                if bond_num == 1:
                    candidate += str(bond_num) + " " + bond + " bond."
                else:
                    candidate += str(bond_num) + " " + bond + " bonds."
            else:
            
                temp_bonds = sample_with_weights(bonds_type, bonds_type_weights, other_bonds)

                temp_bonds_num = []
                for j in range(other_bonds):
                    if temp_bonds[j] == "aromatic":
                        temp_bonds_num.append(random.choices([5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [5, 20, 5, 5, 5, 1, 5, 1, 5, 1, 1, 1, 1, 1, 1], k=1)[0])
                    elif temp_bonds[j] == "single":
                        temp_bonds_num.append(random.randint(1, 100))
                    else:
                        temp_bonds_num.append(random.choices(bonds_num, bonds_num_weights, k=1)[0])
                
                for j in range(len(temp_bonds)):
                    if j == other_bonds - 1:
                        if temp_bonds_num[j] == 1:
                            candidate += "and " + str(temp_bonds_num[j]) + " " + temp_bonds[j] + " bond."
                        else:
                            candidate += "and " + str(temp_bonds_num[j]) + " " + temp_bonds[j] + " bonds."
                    else:
                        if temp_bonds_num[j] == 1:
                            candidate += str(temp_bonds_num[j]) + " " + temp_bonds[j] + " bond, "
                        else:
                            candidate += str(temp_bonds_num[j]) + " " + temp_bonds[j] + " bonds, "
                    temp_bonds_dict[temp_bonds[j]] = temp_bonds_num[j]

            if candidate not in instructions["Instruction"]:
                instructions["Instruction"].append(candidate)
                for key in temp_bonds_dict.keys():
                    instructions[key].append(temp_bonds_dict[key])
                i += 1
        df = pd.DataFrame(instructions)
        df.to_csv(file_dir + "/test.csv", index=False)
elif args.task == "MolOpt":
    if args.subtask == 'LogP':
        prompt_templates = ["Please optimize the molecule {} to have a lower LogP value.", "Modify the molecule {} to decrease its LogP value.", "Optimize the molecule {} to have a lower LogP value.", "Please modify the molecule {} to decrease its LogP value.", "Modify the molecule {} to have a lower LogP value.",
                            "Please optimize the molecule {} to have a higher LogP value.", "Modify the molecule {} to increase its LogP value.", "Optimize the molecule {} to have a higher LogP value.", "Please modify the molecule {} to increase its LogP value.", "Modify the molecule {} to have a higher LogP value."]
        data = pd.read_csv(file_dir + '/test_raw.csv')
        Instructions = {"index":[], "Instruction":[], "molecule":[], "logP":[]}
        for i in range(len(data)):
            molecule = data.iloc[i]['smiles']
            index = data.iloc[i]['index']
            logP = mol_prop(molecule, "logP")
            molecule = molecule.strip().strip('\n').strip()


            text = random.choice(prompt_templates).format(molecule)

            Instructions["index"].append(index)
            Instructions["Instruction"].append(text)
            Instructions["molecule"].append(molecule)
            Instructions["logP"].append(logP)

        df = pd.DataFrame(Instructions)
        df.to_csv(file_dir + "/test.csv", index=False)
    elif args.subtask == 'MR':
        prompt_templates = ["Please optimize the molecule {} to have a lower MR value.", "Modify the molecule {} to decrease its MR value.", "Optimize the molecule {} to have a lower MR value.", "Please modify the molecule {} to decrease its MR value.", "Modify the molecule {} to have a lower MR value.",
                            "Please optimize the molecule {} to have a higher MR value.", "Modify the molecule {} to increase its MR value.", "Optimize the molecule {} to have a higher MR value.", "Please modify the molecule {} to increase its MR value.", "Modify the molecule {} to have a higher MR value."]
        
        data = pd.read_csv(file_dir + '/test_raw.csv')
        Instructions = {"index":[], "Instruction":[], "molecule":[], "MR":[]}
        for i in range(len(data)):
            molecule = data.iloc[i]['smiles']
            index = data.iloc[i]['index']
            MR = mol_prop(molecule, "MR")
            molecule = molecule.strip().strip('\n').strip()

            text = random.choice(prompt_templates).format(molecule)

            Instructions["index"].append(index)
            Instructions["Instruction"].append(text)
            Instructions["molecule"].append(molecule)
            Instructions["MR"].append(MR)
        df = pd.DataFrame(Instructions)
        df.to_csv(file_dir + "/test.csv", index=False)
    elif args.subtask == 'QED':
        prompt_templates = ["Please optimize the molecule {} to have a higher QED value.", "Modify the molecule {} to increase its QED value.", "Optimize the molecule {} to have a higher QED value.", "Please modify the molecule {} to increase its QED value.", "Modify the molecule {} to have a higher QED value.",
                            "Please optimize the molecule {} to have a lower QED value.", "Modify the molecule {} to decrease its QED value.", "Optimize the molecule {} to have a lower QED value.", "Please modify the molecule {} to decrease its QED value.", "Modify the molecule {} to have a lower QED value."]
        data = pd.read_csv(file_dir + '/test_raw.csv')
        Instructions = {"index":[], "Instruction":[], "molecule":[], "QED":[]}
        for i in range(len(data)):
            molecule = data.iloc[i]['smiles']
            index = data.iloc[i]['index']
            QED = mol_prop(molecule, "qed")
            molecule = molecule.strip().strip('\n').strip()

            text = random.choice(prompt_templates).format(molecule)

            Instructions["index"].append(index)
            Instructions["Instruction"].append(text)
            Instructions["molecule"].append(molecule)
            Instructions["QED"].append(QED) 

        df = pd.DataFrame(Instructions)
        df.to_csv(file_dir + "/test.csv", index=False)
    # elif args.subtask == 'TPSA':
    #     pass
        
elif args.task == "InstructionTuning":
    pass
elif args.task == "PPO":
    pass