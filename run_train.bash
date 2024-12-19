CUDA_VISIBLE_DEVICES=1 python instruction_tuning.py --model "meta-llama/Llama-3.2-1B-Instruct" --name llama3.2-1B --data_scale light --num_epochs 2 --save_interval 1000 --warm_up_steps 1000 --disable_lora --train_on_inputs --add_eos
CUDA_VISIBLE_DEVICES=1 python instruction_tuning.py --model "meta-llama/Llama-3.2-1B-Instruct" --name llama3.2-1B --data_scale small --num_epochs 2 --save_interval 1000 --warm_up_steps 1000 --disable_lora --train_on_inputs --add_eos

CUDA_VISIBLE_DEVICES=1 python instruction_tuning.py --model "meta-llama/Llama-3.2-1B-Instruct" --name llama3.2-1B --data_scale medium --num_epochs 2 --save_interval 1000 --warm_up_steps 1000 --disable_lora --train_on_inputs --add_eos
CUDA_VISIBLE_DEVICES=1 python instruction_tuning.py --model "meta-llama/Llama-3.2-1B-Instruct" --name llama3.2-1B --data_scale large --num_epochs 2 --save_interval 1000 --warm_up_steps 1000 --disable_lora --train_on_inputs --add_eos

# CUDA_VISIBLE_DEVICES=1 python instruction_tuning.py --model "meta-llama/Llama-3.2-1B-Instruct" --name llama3.2-1B --data_scale xlarge --num_epochs 2 --save_interval 10000 --warm_up_steps 1000 --disable_lora --train_on_inputs --add_eos

# CUDA_VISIBLE_DEVICES=1 python instruction_tuning.py --model "meta-llama/Llama-3.2-1B-Instruct" --name llama3.2-1B-full --data_scale light --num_epochs 10 --save_interval 1000 --warm_up_steps 1000 --disable_lora


# CUDA_VISIBLE_DEVICES=6 python instruction_tuning.py --model "facebook/galactica-125m" --name galactica-125M --data_scale light --num_epochs 10 --save_interval 1000 --warm_up_steps 100 --disable_lora --specific_task "MR" --train_on_inputs
# CUDA_VISIBLE_DEVICES=6 python test.py --model_type "decoder-only" --name "galactica-125M-light" --base_model "facebook/galactica-125m" --adapter_path "./ckp/galactica-125M-light/checkpoint-150/" --task "MolOpt" --subtask "MR" --output_dir "./independent_predictions/"

# CUDA_VISIBLE_DEVICES=6 python instruction_tuning.py --model "facebook/galactica-125m" --name galactica-125M --data_scale light --num_epochs 10 --save_interval 1000 --warm_up_steps 100 --disable_lora --specific_task "QED" --train_on_inputs
# CUDA_VISIBLE_DEVICES=6 python test.py --model_type "decoder-only" --name "galactica-125M-light" --base_model "facebook/galactica-125m" --adapter_path "./ckp/galactica-125M-light/checkpoint-150/" --task "MolOpt" --subtask "QED" --output_dir "./independent_predictions/"

# CUDA_VISIBLE_DEVICES=7 python instruction_tuning.py --model "facebook/galactica-125m" --name galactica-125M --data_scale small --num_epochs 10 --save_interval 1000 --warm_up_steps 1000 --disable_lora --train_on_inputs

# CUDA_VISIBLE_DEVICES=7 python instruction_tuning.py --model "facebook/galactica-125m" --name galactica-125M --data_scale medium --num_epochs 10 --save_interval 1000 --warm_up_steps 1000 --disable_lora --train_on_inputs


# CUDA_VISIBLE_DEVICES=5 python instruction_tuning.py --model "facebook/galactica-125m" --name galactica-125M --data_scale xlarge --num_epochs 2 --save_interval 1000 --warm_up_steps 1000 --disable_lora --train_on_inputs --scheduler "constant"
