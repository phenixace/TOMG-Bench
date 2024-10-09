import random
import pandas as pd

random.seed(42)

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
        temp_groups = random.choices(groups, groups_weights, k=other_groups)
        temp_groups_num = random.choices(groups_num, groups_num_weights, k=other_groups)
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

# i = 0
# while i < 2000:
#     candidate = random.choice(prompt_templates)

# random.shuffle(instructions["Instruction"])

df = pd.DataFrame(instructions)
df.to_csv("test.csv", index=False)