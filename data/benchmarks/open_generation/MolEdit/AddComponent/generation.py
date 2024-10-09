import random
import pandas as pd

# examine functional groups in the molecule
FunctionalGroups = ["benzene ring", "hydroxyl", "aldehyde", "carboxyl", "amide", "amine", "nitro", "halo", "nitrile", "thiol"]
groups_weights = [15, 15, 5, 5, 10, 5, 5, 5, 1, 1]

prompt_templates = ["Please add a {} to the molecule {}.", "Modify the molecule {} by adding a {}.", "Add a {} to the molecule {}."]

Instructions = {"index":[], "Instruction":[], "molecule":[]}

data = pd.read_csv('./test_raw.csv')
for i in range(len(data)):
    molecule = data.iloc[i]['smiles']
    index = data.iloc[i]['index']

    molecule = molecule.strip().strip('\n').strip()
    
    # randomly select a functional group to add
    to_add = random.choices(FunctionalGroups, groups_weights, k=1)[0]
    text = random.choice(prompt_templates).format(to_add, molecule)

    Instructions["index"].append(index)
    Instructions["Instruction"].append(text)
    Instructions["molecule"].append(molecule)

df = pd.DataFrame(Instructions)
df.to_csv("test.csv", index=False)
