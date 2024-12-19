model="meta-llama/Llama-3.2-1B-Instruct"
name="llama3.2-1B"
data_scale="large"
ckp=5624
gpu=3

CUDA_VISIBLE_DEVICES=$gpu python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolCustom" --subtask "AtomNum" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=$gpu python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolCustom" --subtask "BondNum" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=$gpu python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolCustom" --subtask "FunctionalGroup" --output_dir "./predictions/"

CUDA_VISIBLE_DEVICES=$gpu python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolEdit" --subtask "AddComponent" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=$gpu python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolEdit" --subtask "DelComponent" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=$gpu python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolEdit" --subtask "SubComponent" --output_dir "./predictions/"

CUDA_VISIBLE_DEVICES=$gpu python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolOpt" --subtask "LogP" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=$gpu python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolOpt" --subtask "QED" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=$gpu python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolOpt" --subtask "MR" --output_dir "./predictions/"
