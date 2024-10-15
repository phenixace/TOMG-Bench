'''
For fine-tuning inference
'''
from openai import OpenAI
import os
import re
import json
import rdkit
import argparse
import pandas as pd
from tqdm import tqdm
from utils.dataset import OMGDataset, TMGDataset


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="quantized_models/llama3-70b/")
parser.add_argument("--name", type=str, default="llama3-70B")
parser.add_argument("--port", type=int, default=8000)
# dataset settings
parser.add_argument("--benchmark", type=str, default="open_generation")
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="AtomNum")


parser.add_argument("--output_dir", type=str, default="./predictions/")

parser.add_argument("--temperature", type=float, default=0.75)
parser.add_argument("--top_p", type=float, default=0.85)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=512)

parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--json_check", action="store_true", default=False)
# parser.add_argument("--smiles_check", action="store_true", default=False)

args = parser.parse_args()


