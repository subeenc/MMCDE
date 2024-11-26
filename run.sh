#!/bin/bash
# bash run.sh "dataset"

gpuno="0, 1"
IFS=',' read -ra ADDR <<< "$gpuno"
n_gpu=${#ADDR[@]}

echo "Using ${n_gpu} GPU with DDP Training."
CUDA_VISIBLE_DEVICES=${gpuno}

dataset=$1 

stage='train'
num_epochs=20
train_batch_size=5
dev_batch_size=10
test_batch_size=10
eval_interval=100
local_loss_rate=0.2

torchrun --standalone --nnodes=1 --nproc_per_node=${n_gpu} mmcde_run.py \
    --stage ${stage}\
    --num_train_epochs ${num_epochs} \
    --train_batch_size ${train_batch_size} \
    --dev_batch_size ${dev_batch_size} \
    --test_batch_size ${test_batch_size} \
    --data_dir "${data_dir}" \
    --dataset_name "${dataset_name}" \
    --img_dir "${img_dir}" \
    --image_tensor_dir "${image_tensor_dir}" \
    --init_checkpoint "${init_checkpoint}" \
    --eval_interval ${eval_interval}\
    --local_loss_rate ${local_loss_rate}\
    --log_file "1125_mmcde_train_${dataset_name}_ablation_only_global_view.log"\
    --best_model "best_model_1125_mmcde_train_${dataset_name}_ablation_only_global_view.pt"\
    # --backbone "dialclip"