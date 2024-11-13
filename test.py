'''
For fine-tuning inference
'''
import os
import re
import json
import random
import rdkit
import torch
import argparse
import transformers
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, GenerationConfig
from utils.dataset import OMGDataset, TMGDataset, OMGInsTDataset

from peft import (
    PeftModel
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",  type=str, default="t5")
    parser.add_argument("--name",  type=str, default="molt5-large")
    parser.add_argument("--base_model", type=str, default="laituan245/molt5-large-caption2smiles") # meta-llama/Meta-Llama-3-8B-Instruct
    parser.add_argument("--adapter_path", type=str, default="laituan245/molt5-large-caption2smiles")
    
    parser.add_argument("--benchmark", type=str, default="open_generation")
    parser.add_argument("--task", type=str, default="MolCustom")
    parser.add_argument("--subtask", type=str, default="AtomNum")

    parser.add_argument("--output_dir", type=str, default="./new_predictions/")

    parser.add_argument("--cutoff_len", type=int, default=768)  # anyway, reserve 256 tokens for generation (batch infer)

    # generation config
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    
    parser.add_argument("--seed", type=int, default=42)
    
    # partition inference
    parser.add_argument("--partition", type=int, default=1)
    parser.add_argument("--cur", type=int, default=1)

    parser.add_argument("--enable_lora", default=False, action="store_true")
    parser.add_argument("--int4", default=False, action="store_true")
    parser.add_argument("--int8", default=False, action="store_true")
    parser.add_argument("--fp16", default=False, action="store_true")
    # parser.add_argument("--bf16", default=False, action="store_true")   # in case someone wants to use the bf16 option
    parser.add_argument("--selfies", default=False, action="store_true")
    parser.add_argument("--smiles_check", default=False, action="store_true")
    

    args = parser.parse_args()
        
    # check out put dir
    args.output_dir = args.output_dir + "/" + args.name + "/" + args.benchmark + "/" + args.task + "/"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    
    if os.path.exists(args.output_dir + "/" + args.subtask + "_" + str(args.cur) + ".csv"):
        temp = pd.read_csv(args.output_dir + "/" + args.subtask + "_" + str(args.cur) + ".csv")
        start_pos = len(temp)
    else:
        with open(args.output_dir + args.subtask + "_" + str(args.cur) + ".csv", "w+") as f:
            f.write("outputs\n")
        start_pos = 0
    print("Start from: ", start_pos)

    # set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    transformers.set_seed(args.seed)
    random.seed(args.seed)

    # print parameters
    print("========Parameters========")
    for attr, value in args.__dict__.items():
        print("{}={}".format(attr.upper(), value))

    # load dataset
    if args.benchmark == "open_generation":
        if args.model_type == "t5":
            if args.selfies:
                test_dataset = OMGDataset(args.task, args.subtask, use_selfies=True)
            else:
                test_dataset = OMGDataset(args.task, args.subtask)
        else:
            test_dataset = OMGInsTDataset(args.task, args.subtask)
    elif args.benchmark == "targeted_generation":
        test_dataset = TMGDataset(args.task, args.subtask)
    else:
        raise ValueError("Invalid benchmark: {}".format(args.benchmark))
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    
    # load model
    device_map = "auto"
    if args.enable_lora:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, load_in_8bit=True if args.int8 else False, load_in_4bit=True if args.int4 else False, torch_dtype=torch.float16 if args.fp16 else torch.float32, device_map=device_map)
        model = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype=torch.float16 if args.fp16 else torch.float32, device_map=device_map)    
    else:
        if args.model_type == "decoder-only":
            model = AutoModelForCausalLM.from_pretrained(args.adapter_path, load_in_4bit=True if args.int4 else False, load_in_8bit=True if args.int8 else False, torch_dtype=torch.float16 if args.fp16 else torch.float32, device_map=device_map)
        elif args.model_type == "t5":   # for molt5 and biot5
            model = T5ForConditionalGeneration.from_pretrained(args.adapter_path, device_map=device_map)

    model.eval() 

    generation_config = GenerationConfig(
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                pad_token_id=0,
            )

    # evaluate
    with tqdm(total=int(len(test_dataset)*args.cur/args.partition)-int(len(test_dataset)*(args.cur-1)/args.partition)-start_pos) as pbar:
        pbar.set_description("Inference")
        for idx in range(int(len(test_dataset)*(args.cur-1)/args.partition)+start_pos, int(len(test_dataset)*args.cur/args.partition)):
            error_count = 0
            while True:
                if args.model_type == "t5":
                    text_input = test_dataset.instructions[idx]
                else:
                    text_input = test_dataset.data[idx]
                print(text_input)
                
                model_input = tokenizer(text_input, return_tensors="pt")["input_ids"].cuda()
                # labels = tokenizer(test_data[idx][1], return_tensors="pt")
                with torch.no_grad():
                    # print(model_input) 
                    generation = model.generate(
                        inputs=model_input,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=args.max_new_tokens,
                        num_return_sequences=args.num_return_sequences
                    )

                
                if args.subtask in ["bbbp-uni", "bace-uni", "clintox-uni", "hiv-uni", "toxcast-uni", "sider-uni", "tox21-uni"]:
                        yes_token_id = tokenizer.convert_tokens_to_ids("▁Yes") # mistral
                        no_token_id = tokenizer.convert_tokens_to_ids("▁No")

                        yes_token_id = tokenizer.convert_tokens_to_ids("ĠYes") # llama3
                        no_token_id = tokenizer.convert_tokens_to_ids("ĠNo")

                        s = tokenizer.decode(generation.sequences[0], skip_special_tokens=True)
                        scores = generation.scores[0].softmax(dim=-1)
                        logits = torch.tensor(scores[:,[yes_token_id, no_token_id]], dtype=torch.float32).softmax(dim=-1)[0]
                        s += "\t" + str(logits[0].item())
                else:
                    s = tokenizer.decode(generation.sequences[0], skip_special_tokens=True)
                    if args.model_type == "decoder_only":
                        s = s.split("## Molecule:")[1]

                    if args.smiles_check:
                        try:
                            mol = rdkit.Chem.MolFromSmiles(s.strip().strip("\n").strip())
                        except:
                            error_count += 1
                            print("Error: ", s + " can not be converted to mol")
                            if error_count > 10:
                                break
                    else:
                        break
                    
                
            print(s)
            df = pd.DataFrame([s.strip()], columns=["outputs"])
            df.to_csv(args.output_dir + "/" + args.subtask + "_" + str(args.cur) + ".csv", mode='a', header=False, index=True)
            pbar.update(1)


if __name__ == "__main__":
    main()