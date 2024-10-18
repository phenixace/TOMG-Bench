from argparse import ArgumentParser
from utils.dataset import InsTDataset

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="facebook/galactica-125m")
parser.add_argument("--task", type=str, default="instruction_tuning")

args = parser.parse_args()

