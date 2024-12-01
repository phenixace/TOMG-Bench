'''
For vllm query inference
'''
from openai import OpenAI
import os
import re
import json
import rdkit
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from utils.dataset import OMGDataset, TMGDataset
import transformers
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model, infer_auto_device_map

from accelerate import init_empty_weights
import sys
from accelerate.utils import get_balanced_memory
from torch.cuda.amp import autocast
from torch.utils._python_dispatch import TorchDispatchMode
from dataclasses import dataclass
from typing import Any
import torch.cuda
import multiprocessing as mp

from accelerate import Accelerator

from accelerate.utils import gather_object



parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="quantized_models/llama3-70b/")
parser.add_argument("--name", type=str, default="llama3-70B")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--load_lora", type=bool, default=False)
parser.add_argument("--lora_model_path", type=str, default="")
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
parser.add_argument("--smiles_check", action="store_true", default=False)

# add a log option to record the output
parser.add_argument("--log", action="store_true", default=False)

args = parser.parse_args()

if "mistral" in args.model:
        args.mistral = True
else:
    args.mistral = False

# print parameters
print("========Parameters========")
for attr, value in args.__dict__.items():
    print("{}={}".format(attr.upper(), value))

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://10.140.24.31:{}/v1".format(args.port)

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


# check out put dir
args.output_dir = args.output_dir + args.name + "/" + args.benchmark + "/" + args.task + "/"
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if os.path.exists(args.output_dir + args.subtask + ".csv"):
    temp = pd.read_csv(args.output_dir + args.subtask + ".csv")
    start_pos = len(temp)
else:
    with open(args.output_dir + args.subtask + ".csv", "w+") as f:
        f.write("outputs\n")
    start_pos = 0

print("========Inference Init========")
print("Inference starts from: ", start_pos)


# load dataset
if args.benchmark == "open_generation":
    inference_dataset = OMGDataset(args.task, args.subtask, args.json_check)
elif args.benchmark == "targeted_generation":
    inference_dataset = TMGDataset(args.task, args.subtask, args.json_check)
print("========Sanity Check========")
print(inference_dataset[0])
print("Total length of the dataset:", len(inference_dataset))
print("==============================")

error_records = []

"""
@dataclass
class _ProfilerState:
    cls: Any
    object: Any = None

class TorchDumpDispatchMode(TorchDispatchMode):
    def __init__(self,parent):
        super().__init__()
        self.parent=parent
        self.op_index=0
        self.cvt_count=0

    def get_max_gpu_id(self,tensors):
        max_gpu_id = -1
        max_index = -1
        tensor_index=[]
        for i, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                continue
            tensor_index.append(i)
            if tensor.is_cuda:
                gpu_id = tensor.get_device()
                if gpu_id > max_gpu_id:
                    max_gpu_id = gpu_id
                    max_index = i
        if max_gpu_id == -1:
            return None, None,tensor_index
        return max_index, max_gpu_id,tensor_index

    def convert(self,op_type,tensor_list):
        index, gpu_id,tensor_index = self.get_max_gpu_id(tensor_list)
        if index is None:
            return
        keep_index=set(tensor_index)-set([index])
        device=torch.device(f"cuda:{gpu_id}")
        for i in keep_index:
            if tensor_list[i].device!=device:
                #print(f"{op_type} {i} {tensor_list[i].device} -> {device}")
                tensor_list[i].data=tensor_list[i].data.to(device,non_blocking=True)
                #卡间通信是串行的,所有多stream并不能充分提升性能

    def __torch_dispatch__(self, func, types, args=(),kwargs=None):
        func_packet = func._overloadpacket
        if kwargs is None:
            kwargs = {}
        op_type=f"{func}"
        self.op_index+=1
        if isinstance(args, list) or isinstance(args, tuple):
            self.convert(op_type,args)
        elif isinstance(args[0], list) or isinstance(args[0], tuple):
            self.convert(op_type,args[0])
        else:
            print(op_type)
        output= func(*args,**kwargs)
        return output


class TorchDumper:
    def __init__(self,**kwargs):
        self.p= _ProfilerState(TorchDumpDispatchMode)
        self.kwargs=kwargs

    def __enter__(self):
        if self.p.object is None:
            o = self.p.cls(self,**self.kwargs)
            o.__enter__()
            self.p.object = o
        else:
            self.p.object.step()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        TorchDumper._CURRENT_Dumper = None
        if self.p.object is not None:
            self.p.object.__exit__(exc_type, exc_val, exc_tb)
            del self.p.object
"""

#device = torch.device('cuda')

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size > 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    #gradient_accumulation_steps = gradient_accumulation_steps // world_size

if not ddp and torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True


if args.load_lora == True:
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    #device_map = device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager", trust_remote_code=True).eval()
    print(f"Loading LoRA weights from {args.lora_model_path}")
    model = PeftModel.from_pretrained(model, args.lora_model_path)
    print(f"Merging weights")
    model = model.merge_and_unload()
    print('Convert to BF16...')
    model = model.to(torch.bfloat16)

    accelerator = Accelerator()
    accelerator.prepare(model)

    """
    device_map = infer_auto_device_map(
                    model,
                    dtype='bfloat16')

    model = dispatch_model(model, device_map=device_map)
    """
    #model = dispatch_model(model, device_map=device_map)

else:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map, 
    ).eval()


pipeline = transformers.pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer, 
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        temperature=args.temperature,
        trust_remote_code=True,
        top_p=args.top_p,
)


with tqdm(total=len(inference_dataset)-start_pos) as pbar:
    for idx in range(start_pos, len(inference_dataset)):      
        accelerator.wait_for_everyone()
        cur_seed = args.seed
        error_allowance = 0
        while True:
            #try:
            """
                    completion = client.chat.completions.create(
                        model=args.model,
                        messages=inference_dataset[idx],
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        n=args.num_return_sequences,
                        stop=["</s>", "<|end_of_text|>", "<|eot_id|>"],
                        seed=cur_seed
                    )
                    s = completion.choices[0].message.content
            """
                    
            prompt = inference_dataset[idx]
            inputs = tokenizer.apply_chat_template(prompt,
                                            add_generation_prompt=True,
                                            tokenize=True,
                                            return_tensors="pt",
                                            return_dict=True
                                            )
            inputs = inputs.to(model.device)
            gen_kwargs = {"max_length": args.max_new_tokens, "do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
            #outputs = pipeline(prompt, max_new_tokens=args.max_new_tokens)
            #with TorchDumper
            s_lst = []
            with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    #s = outputs
                    s = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    s_lst.append(s)
                
            s_lst = gather_object(s_lst) #important for gather result to the main gpu
            s = s_lst[0]
            
            """
            except:
                # change random seed
                cur_seed += 1
                error_allowance += 1
                if error_allowance > 10:
                    s = "None"   # empty string
                    error_records.append(idx)
                    break
                else:
                    continue
            """
            
            s = s.replace('""', '"').strip()
            print("Raw:", s)

            if s == None:
                cur_seed += 1
                error_allowance += 1
                if error_allowance > 10:
                    s = ""   # empty string
                    error_records.append(idx)
                    break
                else:
                    continue

            if args.log:
                with open(args.output_dir + args.subtask + ".log", "a+") as f:
                    f.write(s.replace('\n', ' ').strip() + "\n")

            if args.json_check:
                match = re.search(r'\{.*?\}', s, re.DOTALL)
                if match:
                    json_str = match.group()
                    try:
                        json_obj = json.loads(json_str)
                        s = json_obj["molecule"]
                        # add smiles check
                        if args.smiles_check:
                            try:
                                mol = Chem.MolFromSmiles(s)
                                if mol is None:
                                    cur_seed += 1
                                    error_allowance += 1
                                    if error_allowance > 10:
                                        error_records.append(idx)
                                        break
                                    else:
                                        continue
                            except:
                                cur_seed += 1
                                error_allowance += 1
                                if error_allowance > 10:
                                    error_records.append(idx)
                                    break
                                else:
                                    continue
                        break
                    except:
                        # change random seed
                        cur_seed += 1
                        error_allowance += 1
                        if error_allowance > 10:
                            error_records.append(idx)
                            break
                        else:
                            continue

                else:
                    # change random seed
                    cur_seed += 1
                    error_allowance += 1
                    if error_allowance > 10:
                        error_records.append(idx)
                        break
                    else:
                        continue
            else:
                break
        print("Checked:", s)
        
        # check again
        if not isinstance(s, str):
            s = str(s)

        s = s.replace('\n', ' ').strip()  # remove newline characters

        df = pd.DataFrame([s.strip()], columns=["outputs"])
        df.to_csv(args.output_dir +  args.subtask + ".csv", mode='a', header=False, index=True)
        # with open(args.output_dir + "/output_" + args.task + ".txt", "a+") as f:
        #     f.write(s.replace('\n', ' ').strip() + "\n")
        pbar.update(1)


print("========Inference Done========")
print("Error Records: ", error_records)
