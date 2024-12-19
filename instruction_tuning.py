import os
import torch
import random
import argparse
import transformers
from datasets import Dataset
from argparse import ArgumentParser
from utils.dataset import InsTDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="facebook/galactica-125m")
parser.add_argument("--name", type=str, default="galactica-125m")
parser.add_argument("--task", type=str, default="instruction_tuning")
parser.add_argument("--data_scale", type=str, default="large")
parser.add_argument("--output_dir", type=str, default="./ckp/")

# training parameters
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--micro_batch_size", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=1000)
parser.add_argument("--logging_steps", type=int, default=10)
parser.add_argument("--warm_up_steps", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--cutoff_len", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)

# lora parameters
parser.add_argument("--lora_r", type=int, default=64)
parser.add_argument("--lora_alpha", type=int, default=128)
parser.add_argument("--lora_dropout", type=float, default=0.1)

parser.add_argument("--train_on_inputs", default=False, action="store_true")
parser.add_argument("--disable_lora", default=False, action="store_true")
parser.add_argument("--int8", default=False, action="store_true")
parser.add_argument("--fp16", default=False, action="store_true")
parser.add_argument("--add_eos", default=True, action="store_false")
parser.add_argument("--specific_task", type=str, default="")
parser.add_argument("--scheduler", type=str, default="linear")
args = parser.parse_args()

args.output_dir = os.path.join(args.output_dir, args.name + "-" + args.data_scale)
args.specific_task = None if args.specific_task == "" else args.specific_task
args.add_special_token = True if "galactica" in args.name else False
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
print("==========================")

gradient_accumulation_steps = args.batch_size // args.micro_batch_size
# load dataset
if "galactica" in args.name or "mistral" in args.name:
    args.add_eos = "</s>"
else:
    args.add_eos = "<|end_of_text|>"
train_data = InsTDataset(args.data_scale, args.add_eos, args.specific_task, args.add_special_token)
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

train_data = Dataset.from_dict({"gt": train_data.targets, "raw": train_data.data})

print("========Sanity Check========")
for i in range(10):
    print(train_data[i]['gt'])
print("============================")

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference
def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=args.cutoff_len,
        padding=False,
        return_tensors=None,
    )

    result["labels"] = result["input_ids"].copy()

    return result
    
def generate_and_tokenize_prompt(data_point):
    tokenized_full_prompt = tokenize(data_point['gt'])
    if not args.train_on_inputs:
        user_prompt = data_point['raw']
        tokenized_user_prompt = tokenize(user_prompt)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
        
    return tokenized_full_prompt

train_data = (
    train_data.map(lambda sample: generate_and_tokenize_prompt(sample))
)

# load model
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size > 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // world_size

model = AutoModelForCausalLM.from_pretrained(args.model, load_in_8bit=True if args.int8 else False, torch_dtype=torch.float16 if args.fp16 else torch.float32, device_map=device_map)
if args.int8:
    model = prepare_model_for_kbit_training(model)


if not args.disable_lora:
    def generate_peft_config(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        modules = list(lora_module_names)
        
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        return peft_config
    
    config = generate_peft_config(model)
    model = get_peft_model(model, config)

if not args.disable_lora:
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

if not ddp and torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

print("========Sanity Check========")
print(train_data[0])
print("============================")

train_args = TrainingArguments(
    per_device_train_batch_size=args.micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=args.warm_up_steps,
    num_train_epochs=args.num_epochs,
    learning_rate=args.learning_rate,
    fp16=True if args.fp16 else False,
    logging_steps=args.logging_steps,
    optim="adamw_torch",
    eval_strategy="no",
    save_strategy="steps",
    eval_steps=None,
    save_steps=args.save_interval,
    output_dir=args.output_dir,
    save_total_limit=20,
    lr_scheduler_type=args.scheduler,
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False if ddp else None,
    # group_by_length=args.group_by_length,
    report_to="wandb",
    run_name="omg-{}".format(random.randint(0, 100000)),
)

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=None,
    args= train_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

model.config.use_cache = False
    
trainer.train()