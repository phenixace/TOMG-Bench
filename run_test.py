from argparse import ArgumentParser
from utils.dataset import OMGDataset, TMGDataset

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="facebook/galactica-125m")
parser.add_argument("--task", type=str, default="open_generation")
args = parser.parse_args()

