import random

# examine functional groups in the molecule
FunctionalGroups = []
groups_map = {}
temp_groups = []
for group in FunctionalGroups:
    # examine if this functional group exists the molecule
    # if exist
    temp_groups.append(group)

# random select a functional group to substitute
to_sub = random.choice(temp_groups)

text = "Please substitute the functional group {} with the functional group {} in the molecule".format(to_sub, groups_map[to_sub], molecule)

print(text)
