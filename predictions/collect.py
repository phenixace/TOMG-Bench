import pandas as pd
from argparse import ArgumentParser



parser = ArgumentParser()
parser.add_argument("--name", type=str, default="galactica-125M")
parser.add_argument("--benchmark", type=str, default="open_generation")
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="BondNum")
parser.add_argument("--data_scale", type=str, default="medium")
parser.add_argument("--partition", type=int, default=1)

args = parser.parse_args()

file_path = f'./{args.name}-{args.data_scale}/{args.benchmark}/{args.task}/{args.subtask}.csv'

outputs = []
for part in range(1, args.partition+1):
    try:
        temp_data = pd.read_csv(file_path.split('.csv')[0] + f'_{part}.csv')
    except:
        temp_data = pd.read_csv(file_path.split('.csv')[0] + f'_{part}.csv', engine='python')
    molecules = temp_data["outputs"].tolist()
    for molecule in molecules:
        try:
            outputs.append(molecule.split('## Assistant: ')[-1].strip().strip('\n').strip())
        except:
            outputs.append("")

outputs = {"outputs": outputs}

df = pd.DataFrame(outputs)
df.to_csv(file_path, index=False)
