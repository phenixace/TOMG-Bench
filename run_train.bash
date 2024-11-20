# CUDA_VISIBLE_DEVICES=1 python instruction_tuning.py --model "meta-llama/Llama-3.2-1B-Instruct" --name llama3.2-1B-full --data_scale light --num_epochs 10 --save_interval 1000 --warm_up_steps 1000 --disable_lora
# CUDA_VISIBLE_DEVICES=1 python instruction_tuning.py --model "meta-llama/Llama-3.2-1B-Instruct" --name llama3.2-1B --data_scale small --num_epochs 10 --save_interval 1000 --warm_up_steps 1000

# CUDA_VISIBLE_DEVICES=1 python instruction_tuning.py --model "meta-llama/Llama-3.2-1B-Instruct" --name llama3.2-1B --data_scale medium --num_epochs 10 --save_interval 1000 --warm_up_steps 1000
# CUDA_VISIBLE_DEVICES=1 python instruction_tuning.py --model "meta-llama/Llama-3.2-1B-Instruct" --name llama3.2-1B --data_scale large --num_epochs 10 --save_interval 1000 --warm_up_steps 1000

# CUDA_VISIBLE_DEVICES=1 python instruction_tuning.py --model "meta-llama/Llama-3.2-1B-Instruct" --name llama3.2-1B-full --data_scale light --num_epochs 10 --save_interval 1000 --warm_up_steps 1000 --disable_lora


# CUDA_VISIBLE_DEVICES=7 python instruction_tuning.py --model "facebook/galactica-125m" --name galactica-125M --data_scale light --num_epochs 10 --save_interval 1000 --warm_up_steps 100 --disable_lora --train_on_inputs

CUDA_VISIBLE_DEVICES=7 python instruction_tuning.py --model "facebook/galactica-125m" --name galactica-125M --data_scale small --num_epochs 10 --save_interval 1000 --warm_up_steps 1000 --disable_lora --train_on_inputs

CUDA_VISIBLE_DEVICES=7 python instruction_tuning.py --model "facebook/galactica-125m" --name galactica-125M --data_scale medium --num_epochs 10 --save_interval 1000 --warm_up_steps 1000 --disable_lora --train_on_inputs
