model="facebook/galactica-125m"
name="galactica-125M"
data_scale="light"
ckp=1400

CUDA_VISIBLE_DEVICES=7 python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolCustom" --subtask "AtomNum" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=1 python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolCustom" --subtask "BondNum" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=1 python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolCustom" --subtask "FunctionalGroup" --output_dir "./predictions/"

CUDA_VISIBLE_DEVICES=1 python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolEdit" --subtask "AddComponent" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=1 python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolEdit" --subtask "DelComponent" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=1 python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolEdit" --subtask "SubComponent" --output_dir "./predictions/"

CUDA_VISIBLE_DEVICES=1 python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolOpt" --subtask "LogP" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=1 python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolOpt" --subtask "QED" --output_dir "./predictions/"
CUDA_VISIBLE_DEVICES=1 python test.py --model_type "decoder-only" --name $name-$data_scale --base_model $model --adapter_path ./ckp/$name-$data_scale/checkpoint-$ckp/ --task "MolOpt" --subtask "MR" --output_dir "./predictions/"
