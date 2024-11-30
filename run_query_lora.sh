accelerate launch --main_process_port 29503  query.py --task MolEdit --subtask AddComponent --json_check --port 8002 --name llama-3.1-light --model meta/Llama-3.1-8B --load_lora True --lora_model_path ./ckpt/llama3.1-8B-light/checkpoint-1410
accelerate launch --main_process_port 29503  query.py --task MolEdit --subtask DelComponent --json_check --port 8002 --name llama-3.1-light --model meta/Llama-3.1-8B --load_lora True --lora_model_path ./ckpt/llama3.1-8B-light/checkpoint-1410
accelerate launch --main_process_port 29503  query.py --task MolEdit --subtask SubComponent --json_check --port 8002 --name llama-3.1-light --model meta/Llama-3.1-8B --load_lora True --lora_model_path ./ckpt/llama3.1-8B-light/checkpoint-1410

accelerate launch --main_process_port 29503  query.py --task MolCustom --subtask AtomNum --json_check --port 8002 --name llama-3.1-light --model meta/Llama-3.1-8B --load_lora True --lora_model_path ./ckpt/llama3.1-8B-light/checkpoint-1410
accelerate launch --main_process_port 29503  query.py --task MolCustom --subtask BondNum --json_check --port 8002 --name llama-3.1-light --model meta/Llama-3.1-8B --load_lora True --lora_model_path ./ckpt/llama3.1-8B-light/checkpoint-1410
accelerate launch --main_process_port 29503  query.py --task MolCustom --subtask FunctionalGroup --json_check --port 8002 --name llama-3.1-light --model meta/Llama-3.1-8B --load_lora True --lora_model_path ./ckpt/llama3.1-8B-light/checkpoint-1410

accelerate launch --main_process_port 29503  query.py --task MolOpt --subtask LogP --json_check --port 8002 --name llama-3.1-light --model meta/Llama-3.1-8B --load_lora True --lora_model_path ./ckpt/llama3.1-8B-light/checkpoint-1410
accelerate launch --main_process_port 29503  query.py --task MolOpt --subtask MR --json_check --port 8002 --name llama-3.1-light --model meta/Llama-3.1-8B --load_lora True --lora_model_path ./ckpt/llama3.1-8B-light/checkpoint-1410
accelerate launch --main_process_port 29503  query.py --task MolOpt --subtask QED --json_check --port 8002 --name llama-3.1-light --model meta/Llama-3.1-8B --load_lora True --lora_model_path ./ckpt/llama3.1-8B-light/checkpoint-1410
