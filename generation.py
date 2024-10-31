from argparse import ArgumentParser
import random
import pandas as pd
import copy
from utils.evaluation import mol_prop
import datasets
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--task", type=str, default="InstructionTuning")
parser.add_argument("--subtask", type=str, default="BasicProp")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_samples", type=int, default=90000)
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

def AddComponent(mol):
    # # first extract all the functional groups in the molecule
    groups = ["benzene ring", "hydroxyl", "aldehyde", "carboxyl", "amide", "amine", "nitro", "halo", "nitrile", "thiol"]
    groups_weights = [15, 15, 5, 5, 10, 5, 5, 5, 1, 1]
    groups_SMARTS = ['[cR1]1[cR1][cR1][cR1][cR1][cR1]1', '[OX2H]', '[CX3H1](=O)[#6]', '[CX3](=O)[OX2H1]', '[NX3][CX3](=[OX1])[#6]', '[NX3;H2,H1;!$(NC=O)]', '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]', '[F,Cl,Br,I]', '[NX1]#[CX2]', '[#16X2H]']
    groups_SMARTS_dict = dict(zip(groups, groups_SMARTS))
    
    group = random.choices(groups, groups_weights, k=1)[0]
    
    try:
        rwmol = Chem.RWMol(mol)
    except:
        return None, None
    # # add the functional group to the molecule
    functional_group = Chem.MolFromSmarts(groups_SMARTS_dict[group])

    # # carbon atoms in the molecule
    carbon_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
    # # randomly choose a carbon atom to attach the functional group

    atom_to_attach = random.choice(carbon_atoms)  # 选择一个碳原子作为连接点

    # 创建一个映射，将官能团中原子的索引映射到新分子中的索引
    atom_mapping = {}

    # 添加官能团中的原子到新分子中
    for atom in functional_group.GetAtoms():
        new_atom_idx = rwmol.AddAtom(atom)
        atom_mapping[atom.GetIdx()] = new_atom_idx

    # 添加官能团中的键到新分子中
    for bond in functional_group.GetBonds():
        begin_idx = atom_mapping[bond.GetBeginAtomIdx()]
        end_idx = atom_mapping[bond.GetEndAtomIdx()]
        rwmol.AddBond(begin_idx, end_idx, bond.GetBondType())
    
    rwmol.AddBond(atom_to_attach, new_atom_idx, Chem.BondType.SINGLE)

    new_mol = rwmol.GetMol()
    new_mol = Chem.MolToSmiles(new_mol)
    return new_mol, group

def DelComponent(mol):
    # first extract all the functional groups in the molecule
    groups = ["benzene ring", "hydroxyl", "aldehyde", "carboxyl", "amide", "amine", "nitro", "halo", "nitrile", "thiol"]
    groups_SMARTS = ['[cR1]1[cR1][cR1][cR1][cR1][cR1]1', '[OX2H]', '[CX3H1](=O)[#6]', '[CX3](=O)[OX2H1]', '[NX3][CX3](=[OX1])[#6]', '[NX3;H2,H1;!$(NC=O)]', '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]', '[F,Cl,Br,I]', '[NX1]#[CX2]', '[#16X2H]']
    groups_SMARTS_dict = dict(zip(groups, groups_SMARTS))

    existed_groups = []
    for group in groups:
        if group == "benzene ring":
            group_num = mol_prop(molecule, "num_benzene_ring")
        else:
            group_num = mol_prop(molecule, "num_"+group)
        if group_num > 0:
            existed_groups.append((group, group_num))
    if len(existed_groups) == 0:
        return None, None

    # randomly choose a functional group to remove
    group, group_num = random.choice(existed_groups)

    try:
        rwmol = Chem.RWMol(mol)
    except:
        return None, None
    # find the atoms in the molecule that match the functional group
    matched_atoms = mol.GetSubstructMatches(Chem.MolFromSmarts(groups_SMARTS_dict[group]))
    matched_group = random.choice(matched_atoms)
    # Remove all atoms in the matched functional group
    # It's important to remove atoms in reverse order to avoid index issues

    end_points = set()
    for atom_idx in matched_group:
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            other_idx = bond.GetOtherAtomIdx(atom.GetIdx())
            if other_idx not in matched_group:
                end_points.add(other_idx)

    end_points = list(end_points)
    print(end_points)
    if len(end_points) == 2:
        try:
            rwmol.AddBond(end_points[0], end_points[1], Chem.BondType.SINGLE)
        except:
            pass
    elif len(end_points) > 2:
        for i in range(len(end_points)-1):
            try:
                rwmol.AddBond(end_points[i], end_points[i+1], Chem.BondType.SINGLE)
            except:
                pass

    for atom_idx in sorted(matched_group, reverse=True):
        rwmol.RemoveAtom(atom_idx)


    new_mol = rwmol.GetMol()
    new_mol = Chem.MolToSmiles(new_mol)

    return new_mol, group


def SubComponent(mol):
    # only edge groups can be substituted in our case
    groups = ["hydroxyl", "aldehyde", "carboxyl", "nitro", "halo", "nitrile", "thiol"]
    
    groups_smiles = ['[OX2H]', '[CX3H1](=O)[#6]', '[CX3](=O)[OX2H1]', 'NO', '[F,Cl,Br,I]', 'C#N', '[#16X2H]']
    groups_SMARTS_dict = dict(zip(groups, groups_smiles))
    
    existed_groups = []
    for group in groups:

        group_num = mol_prop(molecule, "num_"+group)
        if group_num > 0:
            existed_groups.append((group, group_num))
    if len(existed_groups) == 0:
        return None, None, None

    # randomly choose a functional group to be substituted
    group, group_num = random.choice(existed_groups)

    # choose a different functional group to substitute
    groups.remove(group)
    new_group = random.choice(groups)
    if new_group == "halo":
        halos = ["[F]", "[Cl]", "[Br]", "[I]"]
        temp_group = random.choice(halos)
        new_mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmarts(groups_SMARTS_dict[group]), Chem.MolFromSmarts(temp_group))
    else:
        new_mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmarts(groups_SMARTS_dict[group]), Chem.MolFromSmarts(groups_SMARTS_dict[new_group]))
    
    new_mol = new_mol[0] if isinstance(new_mol, tuple) else new_mol
    
    
    new_mol = Chem.MolToSmiles(new_mol)
    return new_mol, group, new_group

if args.task == "InstructionTuning":
    file_dir = f'./data/instruction_tuning/'
else:
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
        FunctionalGroups = ["hydroxyl", "aldehyde", "carboxyl", "nitro", "halo", "nitrile", "thiol"]

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
            if carbon_num == 1:
                candidate = random.choice(prompt_templates) + str(carbon_num) + " carbon atom"
            else:
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
                if element_num == 1:
                    candidate += " and " + str(element_num) + " " + element + " atom."
                else:
                    candidate += " and " + str(element_num) + " " + element + " atoms."
                Other_elements_dict[element] = element_num
            else:
                candidate += ", "
                temp_elements = sample_with_weights(elements, elements_weights, other_elements)
                
                temp_elements_num = sample_with_weights(elements_num, elements_num_weights, other_elements)
                for j in range(len(temp_elements)):
                    if j == other_elements - 1:
                        if temp_elements_num[j] == 1:
                            candidate += "and " + str(temp_elements_num[j]) + " " + temp_elements[j] + " atom."
                        else:
                            candidate += "and " + str(temp_elements_num[j]) + " " + temp_elements[j] + " atoms."
                    else:
                        if temp_elements_num[j] == 1:
                            candidate += str(temp_elements_num[j]) + " " + temp_elements[j] + " atom, "
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
        groups = ["benzene rings", "hydroxyl", "anhydride", "aldehyde", "ketone", "carboxyl", "ester", "amide", "amine", "nitro", "halo", "thioether", "nitrile", "thiol", "sulfide", "disulfide", "sulfoxide", "sulfone", "borane"]
        groups_weights = [15, 15, 2, 5, 5, 10, 5, 5, 5, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]

        groups_num = [1, 2, 3, 4, 5]
        groups_num_weights = [10, 5, 2, 1, 1]


        prompt_templates = ["Please generate a molecule with ", "Please generate a molecule composed of ", "Please generate a molecule consisting ", "The molecule has ", "The molecule is composed of ", "The molecule consists of ", "There is a molecule with ", "There is a molecule composed of ", "There is a molecule consisting of ", "The molecule contains "]

        instructions = {"Instruction":[], "benzene rings":[], "hydroxyl":[], "anhydride":[], "aldehyde":[], "ketone":[], "carboxyl":[], "ester":[], "amide":[], "amine":[], "nitro":[], "halo":[], "thioether":[], "nitrile":[], "thiol":[], "sulfide":[], "disulfide":[], "sulfoxide":[], "sulfone":[], "borane":[]}
        i = 0
        while i < 5000:
            candidate = random.choice(prompt_templates)

            other_groups_num = [item for item in range(1, 5)]
            other_groups_num_weights = [int(10/(item)) for item in other_groups_num]
            other_groups = random.choices(other_groups_num, other_groups_num_weights, k=1)[0]

            temp_groups_dict = {"benzene rings":0, "hydroxyl":0, "anhydride":0, "aldehyde":0, "ketone":0, "carboxyl":0, "ester":0, "amide":0, "amine":0, "nitro":0, "halo":0, "thioether":0, "nitrile":0, "thiol":0, "sulfide":0, "disulfide":0, "sulfoxide":0, "sulfone":0, "borane":0}

            if other_groups == 1:
                group = random.choices(groups, groups_weights, k=1)[0]
                group_num = random.choices(groups_num, groups_num_weights, k=1)[0]
                temp_groups_dict[group] = group_num
                if group_num == 1:
                    candidate += str(group_num) + " " + group + " group."
                else:
                    candidate += str(group_num) + " " + group + " groups."
            else:
                temp_groups = sample_with_weights(groups, groups_weights, other_groups)
                temp_groups_num = sample_with_weights(groups_num, groups_num_weights, other_groups)
                for j in range(len(temp_groups)):
                    if j == other_groups - 1:
                        if temp_groups_num[j] == 1:
                            candidate += "and " + str(temp_groups_num[j]) + " " + temp_groups[j] + " group."
                        else:
                            candidate += "and " + str(temp_groups_num[j]) + " " + temp_groups[j] + " groups."
                    else:
                        if temp_groups_num[j] == 1:
                            candidate += str(temp_groups_num[j]) + " " + temp_groups[j] + " group, "
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
    # TODO: extract molecules from the PubChem dataset that are not contained in the zinc dataset
    
    zinc_data = pd.read_csv('./data/sources/zinc250k/zinc250k_selfies.csv')
    existing_molecules = zinc_data['smiles'].tolist()

    pubchem_data = datasets.load_dataset('./data/sources/pubchem10m')
    pubchem_data = pubchem_data['train']
    pubchem_molecules = pubchem_data['smiles']
    tasks = ['AtomNum', 'BondNum', 'FunctionalGroup', 'AddComponent', 'SubComponent', 'DelComponent', 'LogP', 'MR', 'QED']

    data_frame = {"SubTask":[], "Instruction":[], "molecule":[]}
    start_pos = 0
    cur = 0
    with tqdm(total=args.num_samples) as pbar:
        for molecule in pubchem_molecules:
            if cur < start_pos:
                cur += 1
                continue

            if molecule not in existing_molecules:
                print(molecule)
                try:
                    mol = Chem.MolFromSmiles(molecule)
                    if mol is None:
                        continue
                except:
                    continue
                itemtask = tasks[cur % 9]
                if itemtask == 'AtomNum':
                    # generate the instruction for the molecule
                    # - the instruction should be like "Please generate a molecule with 10 carbon atoms
                    elements = ["carbon", "oxygen", "nitrogen", "sulfur", "fluorine", "chlorine", "bromine", "iodine", "phosphorus", "boron", "silicon", "selenium", "tellurium", "arsenic", "antimony", "bismuth", "polonium"]
                    templates = ["Please generate a molecule with ", "Please generate a molecule composed of ", "Please generate a molecule consisting ", "The molecule has ", "The molecule is composed of ", "The molecule consists of ", "There is a molecule with ", "There is a molecule composed of ", "There is a molecule consisting of ", "The molecule contains "]
                    existed_elements = []
                    for element in elements:
                        element_num = mol_prop(molecule, "num_"+element)
                        if element_num > 0:
                            # generate the instruction
                            existed_elements.append((element, element_num))
                    if len(existed_elements) == 0:
                        continue
                    elif len(existed_elements) == 1:
                        template = random.choice(templates)
                        element, element_num = existed_elements[0]
                        if element_num == 1:
                            template += str(element_num) + " " + element + " atom."
                        else:
                            template += str(element_num) + " " + element + " atoms."
                    else:
                        template = random.choice(templates)
                        for i in range(len(existed_elements)):
                            element, element_num = existed_elements[i]
                            if i == len(existed_elements) - 1:
                                if element_num == 1:
                                    template += "and " + str(element_num) + " " + element + " atom."
                                else:
                                    template += "and " + str(element_num) + " " + element + " atoms."
                            else:
                                if element_num == 1:
                                    template += str(element_num) + " " + element + " atom, "
                                else:
                                    template += str(element_num) + " " + element + " atoms, "

                    # save the instruction
                    print(template)
                    data_frame["SubTask"].append(itemtask)
                    data_frame["Instruction"].append(template)
                    data_frame["molecule"].append(molecule)
                elif itemtask == 'BondNum':
                    bonds = ["single", "double", "triple", "rotatable", "aromatic"]
                    templates = ["Please generate a molecule with ", "Please generate a molecule composed of ", "Please generate a molecule consisting ", "The molecule has ", "The molecule is composed of ", "The molecule consists of ", "There is a molecule with ", "There is a molecule composed of ", "There is a molecule consisting of ", "The molecule contains "]
                    existed_bonds = []
                    for bond in bonds:
                        bond_num = mol_prop(molecule, "num_"+bond+"_bonds")
                        if bond_num > 0:
                            existed_bonds.append((bond, bond_num))
                    if len(existed_bonds) == 0:
                        continue
                    elif len(existed_bonds) == 1:
                        template = random.choice(templates)
                        bond, bond_num = existed_bonds[0]
                        if bond_num == 1:
                            template += str(bond_num) + " " + bond + " bond."
                        else:
                            template += str(bond_num) + " " + bond + " bonds."
                    else:
                        template = random.choice(templates)
                        for i in range(len(existed_bonds)):
                            bond, bond_num = existed_bonds[i]
                            if i == len(existed_bonds) - 1:
                                if bond_num == 1:
                                    template += "and " + str(bond_num) + " " + bond + " bond."
                                else:
                                    template += "and " + str(bond_num) + " " + bond + " bonds."
                            else:
                                if bond_num == 1:
                                    template += str(bond_num) + " " + bond + " bond, "
                                else:
                                    template += str(bond_num) + " " + bond + " bonds, "
                    print(template)
                    data_frame["SubTask"].append(itemtask)
                    data_frame["Instruction"].append(template)
                    data_frame["molecule"].append(molecule)
                elif itemtask == 'FunctionalGroup':
                    groups = ["benzene ring", "hydroxyl", "anhydride", "aldehyde", "ketone", "carboxyl", "ester", "amide", "amine", "nitro", "halo", "thioether", "nitrile", "thiol", "sulfide", "disulfide", "sulfoxide", "sulfone", "borane"]
                    templates = ["Please generate a molecule with ", "Please generate a molecule composed of ", "Please generate a molecule consisting ", "The molecule has ", "The molecule is composed of ", "The molecule consists of ", "There is a molecule with ", "There is a molecule composed of ", "There is a molecule consisting of ", "The molecule contains "]
                    existed_groups = []
                    for group in groups:
                        if group == "benzene ring":
                            group_num = mol_prop(molecule, "num_benzene_ring")
                        else:
                            group_num = mol_prop(molecule, "num_"+group)
                        if group_num > 0:
                            existed_groups.append((group, group_num))
                    if len(existed_groups) == 0:
                        continue
                    elif len(existed_groups) == 1:
                        template = random.choice(templates)
                        group, group_num = existed_groups[0]
                        if group_num == 1:
                            template += str(group_num) + " " + group + " group."
                        else:
                            template += str(group_num) + " " + group + " groups."
                    template = random.choice(templates)
                    for i in range(len(existed_groups)):
                        group, group_num = existed_groups[i]
                        if i == len(existed_groups) - 1:
                            if group_num == 1:
                                template += "and " + str(group_num) + " " + group + " group."
                            else:

                                template += "and " + str(group_num) + " " + group + " groups."
                        else:
                            if group_num == 1:
                                template += str(group_num) + " " + group + " group, "
                            else:
                                template += str(group_num) + " " + group + " groups, "
                    print(template)
                    data_frame["SubTask"].append(itemtask)
                    data_frame["Instruction"].append(template)
                    data_frame["molecule"].append(molecule)
                elif itemtask == 'AddComponent':
                    templates = ["Please add a {} to the molecule {}.", "Modify the molecule {} by adding a {}.", "Add a {} to the molecule {}."]
                    try:
                        new_mol, group = AddComponent(mol)
                    except:
                        continue
                    template = random.randint(0, 2)
                    if template == 0 or template == 2:
                        template = templates[template].format(group, molecule)
                    else:
                        template = templates[template].format(molecule, group)
                    
                    print(template)
                    
                    if "." in new_mol:    
                        continue
                    try:
                        mol = Chem.MolFromSmiles(new_mol)
                    except:
                        continue
                    data_frame["SubTask"].append(itemtask)
                    data_frame["Instruction"].append(template)
                    data_frame["molecule"].append(new_mol)
                    print(new_mol)
                    
        
                elif itemtask == 'SubComponent':
                    
                    templates = ["Please substitute a {} in the molecule {} with a {}.", "Modify the molecule {} by substituting a {} with a {}.", "Substitute a {} in the molecule {} with a {}."]
                    try:
                        new_mol, group, new_group = SubComponent(mol)
                    except:
                        continue
                    if new_mol == None:
                        continue
                    template = random.randint(0,2)
                    if template == 0 or template == 2:
                        template = random.choice(templates).format(group, molecule, new_group)
                    else:
                        template = random.choice(templates).format(molecule, group, new_group)
                    print(template)
                    if "." in new_mol:    
                        continue
                    try:
                        mol = Chem.MolFromSmiles(new_mol)
                    except:
                        continue
                    data_frame["SubTask"].append(itemtask)
                    data_frame["Instruction"].append(template)
                    data_frame["molecule"].append(new_mol)
                    print(new_mol)
                    

                elif itemtask == 'DelComponent':
                    
                    templates = ["Please remove a {} from the molecule {}.", "Modify the molecule {} by removing a {}.", "Remove a {} from the molecule {}."]
                    try:
                        new_mol, group = DelComponent(mol)
                    except:
                        continue
                    if new_mol == None:
                        continue
                    
                    template = random.randint(0,2)
                    if template == 0 or template == 2:
                        template = random.choice(templates).format(group, molecule)
                    else:
                        template = random.choice(templates).format(molecule, group)
                    if "." in new_mol:    
                        continue
                    try:
                        mol = Chem.MolFromSmiles(new_mol)
                    except:
                        continue
                    print(template)
                    data_frame["SubTask"].append(itemtask)
                    data_frame["Instruction"].append(template)
                    data_frame["molecule"].append(new_mol)
                    print(new_mol)
    
                elif itemtask == 'LogP':
                    method = random.randint(0, 2)
                    templates_low = ["Please optimize the molecule {} to have a lower LogP value.", "Modify the molecule {} to decrease its LogP value.", "Optimize the molecule {} to have a lower LogP value.", "Please modify the molecule {} to decrease its LogP value.", "Modify the molecule {} to have a lower LogP value."]
                    templates_high = ["Please optimize the molecule {} to have a higher LogP value.", "Modify the molecule {} to increase its LogP value.", "Optimize the molecule {} to have a higher LogP value.", "Please modify the molecule {} to increase its LogP value.", "Modify the molecule {} to have a higher LogP value."]
                    if method == 0:
                        try:
                            new_mol, _ = AddComponent(mol)
                        except:
                            continue
                        if new_mol == None:
                            method += 1
                        elif "." in new_mol:    
                            method += 1
                        try:
                            mol = Chem.MolFromSmiles(new_mol)
                        except:
                            method += 1
                    if method == 1:
                        try:
                            new_mol, _, _ = SubComponent(mol)
                        except:
                            continue
                        if new_mol == None:
                            method += 1
                        elif "." in new_mol:    
                            method += 1
                        try:
                            mol = Chem.MolFromSmiles(new_mol)
                        except:
                            method += 1
                    if method == 2:
                        try:
                            new_mol, _ = DelComponent(mol)
                        except:
                            continue
                        if new_mol == None:
                            continue
                        elif "." in new_mol:    
                            continue
                        try:
                            mol = Chem.MolFromSmiles(new_mol)
                        except:
                            continue
                    print(new_mol)
                    if new_mol == None:
                        continue
                    logP = mol_prop(molecule, "logP")
                    new_logP = mol_prop(new_mol, "logP")
                    if new_logP == None:
                        continue
                    if logP > new_logP:
                        template = random.choice(templates_low).format(molecule)
                    else:
                        template = random.choice(templates_high).format(molecule)

                    print(template)
                    data_frame["SubTask"].append(itemtask)
                    data_frame["Instruction"].append(template)
                    data_frame["molecule"].append(new_mol)


                elif itemtask == 'MR':
                    method = random.randint(0, 2)
                    templates_low = ["Please optimize the molecule {} to have a lower MR value.", "Modify the molecule {} to decrease its MR value.", "Optimize the molecule {} to have a lower MR value.", "Please modify the molecule {} to decrease its MR value.", "Modify the molecule {} to have a lower MR value."]
                    templates_high = ["Please optimize the molecule {} to have a higher MR value.", "Modify the molecule {} to increase its MR value.", "Optimize the molecule {} to have a higher MR value.", "Please modify the molecule {} to increase its MR value.", "Modify the molecule {} to have a higher MR value."]
                    if method == 0:
                        try:
                            new_mol, _ = AddComponent(mol)
                        except:
                            continue
                        if new_mol == None:
                            method += 1
                        elif "." in new_mol:    
                            method += 1
                        try:
                            mol = Chem.MolFromSmiles(new_mol)
                        except:
                            method += 1
                    if method == 1:
                        try:
                            new_mol, _, _ = SubComponent(mol)
                        except:
                            continue
                        if new_mol == None:
                            method += 1
                        elif "." in new_mol:    
                            method += 1
                        try:
                            mol = Chem.MolFromSmiles(new_mol)
                        except:
                            method += 1
                    if method == 2:
                        try:
                            new_mol, _ = DelComponent(mol)
                        except:
                            continue
                        if new_mol == None:
                            continue
                        elif "." in new_mol:    
                            continue
                        try:
                            mol = Chem.MolFromSmiles(new_mol)
                        except:
                            continue
                    print(new_mol)
                    if new_mol == None:
                        continue
                    MR = mol_prop(molecule, "MR")
                    new_MR = mol_prop(new_mol, "MR")
                    if new_MR == None:
                        continue
                    if MR > new_MR:
                        template = random.choice(templates_low).format(molecule)
                    else:
                        template = random.choice(templates_high).format(molecule)

                    print(template)
                    data_frame["SubTask"].append(itemtask)
                    data_frame["Instruction"].append(template)
                    data_frame["molecule"].append(new_mol)

                elif itemtask == 'QED':
                    method = random.randint(0, 2)
                    templates_low = ["Please optimize the molecule {} to have a lower QED value.", "Modify the molecule {} to decrease its QED value.", "Optimize the molecule {} to have a lower QED value.", "Please modify the molecule {} to decrease its QED value.", "Modify the molecule {} to have a lower QED value."]
                    templates_high = ["Please optimize the molecule {} to have a higher QED value.", "Modify the molecule {} to increase its QED value.", "Optimize the molecule {} to have a higher QED value.", "Please modify the molecule {} to increase its QED value.", "Modify the molecule {} to have a higher QED value."]
                    
                    if method == 0:
                        try:
                            new_mol, _ = AddComponent(mol)
                        except:
                            continue
                        if new_mol == None:
                            method += 1
                        elif "." in new_mol:    
                            method += 1
                        try:
                            mol = Chem.MolFromSmiles(new_mol)
                        except:
                            method += 1
                    if method == 1:
                        try:
                            new_mol, _, _ = SubComponent(mol)
                        except:
                            continue
                        if new_mol == None:
                            method += 1
                        elif "." in new_mol:    
                            method += 1
                        try:
                            mol = Chem.MolFromSmiles(new_mol)
                        except:
                            method += 1
                    if method == 2:
                        try:
                            new_mol, _ = DelComponent(mol)
                        except:
                            print(molecule)
                            continue
                        if new_mol == None:
                            continue
                        elif "." in new_mol:    
                            continue
                        try:
                            mol = Chem.MolFromSmiles(new_mol)
                        except:
                            continue
                    print(new_mol)
                    if new_mol == None:
                        continue
                    QED = mol_prop(molecule, "qed")
                    new_QED = mol_prop(new_mol, "qed")
                    if new_QED == None:
                        continue
                    if QED > new_QED:
                        template = random.choice(templates_low).format(molecule)
                    else:
                        template = random.choice(templates_high).format(molecule)

                    print(template)
                    data_frame["SubTask"].append(itemtask)
                    data_frame["Instruction"].append(template)
                    data_frame["molecule"].append(new_mol)
                cur += 1
                pbar.update(1)
                if cur >= args.num_samples:
                    break    
    df = pd.DataFrame(data_frame)
    df.to_csv(file_dir + "/train.csv", index=False)
    
elif args.task == "PPO":
    # TODO: the instruction following is also considered.
    # How reward is built?
    pass