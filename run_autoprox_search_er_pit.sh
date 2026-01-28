#!/usr/bin/env bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="3"

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ERT-NAS

# Run Auto-Prox search with er in search space
# For PiT
# conda run -n ERT-NAS python search_autoprox_with_er.py \
#     --gt_path ./gt_results/gt_pit.pth \
#     --refer_cfg ./configs/auto/pit/pit-ti_c100_kd.yaml \
#     --ds cifar100 \
#     --acc_type kd \
#     --num_generations 3 \
#     --pop_size 5 \
#     --save_dir work_dirs/autoprox_search

# conda run -n ERT-NAS python search_autoprox_with_er.py \
#     --gt_path ./gt_results/gt_pit.pth \
#     --refer_cfg ./configs/auto/pit/pit-ti_flowers_kd.yaml \
#     --ds flowers \
#     --acc_type kd \
#     --num_generations 3 \
#     --pop_size 5 \
#     --save_dir work_dirs/autoprox_search
    
conda run -n ERT-NAS python search_autoprox_with_er.py \
    --gt_path ./gt_results/gt_pit.pth \
    --refer_cfg ./configs/auto/pit/pit-ti_chaoyang_kd.yaml \
    --ds chaoyang \
    --acc_type kd \
    --num_generations 3 \
    --pop_size 5 \
    --save_dir work_dirs/autoprox_search

echo "Auto-Prox search completed!"
