import random
import pandas as pd

random.seed(42)

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
        temp_elements = random.choices(elements, elements_weights, k=other_elements)
        temp_elements_num = random.choices(elements_num, elements_num_weights, k=other_elements)
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
# i = 0
# while i < 500:
#     carbon_num = random.randint(1, 100)
#     candidate = random.choice(prompt_templates) + str(carbon_num) + " atoms."
#     if candidate not in instructions["Instruction"]:
#         instructions["Instruction"].append(candidate)
#         instructions["carbon"].append(carbon_num)
#         for key in Other_elements_dict.keys():
#             instructions[key].append(0)
#         i += 1


df = pd.DataFrame(instructions)
df.to_csv("test.csv", index=False)