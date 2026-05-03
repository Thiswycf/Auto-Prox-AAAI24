#!/usr/bin/env bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ERT-NAS

# Run Auto-Prox search with er in search space
# For AutoFormer
# conda run -n ERT-NAS python search_autoprox_with_er.py \
#     --gt_path ./gt_results/gt_autoformer_2.pth \
#     --refer_cfg ./configs/auto/autoformer/autoformer-ti-subnet_c100_kd.yaml \
#     --ds cifar100 \
#     --acc_type kd \
#     --num_generations 3 \
#     --pop_size 5 \
#     --expected_kt 0.56 \
#     --save_dir work_dirs/autoprox_search
    
conda run -n ERT-NAS python search_autoprox_with_er.py \
    --gt_path ./gt_results/gt_autoformer_2.pth \
    --refer_cfg ./configs/auto/autoformer/autoformer-ti-subnet_chaoyang_kd.yaml \
    --ds chaoyang \
    --acc_type kd \
    --num_generations 3 \
    --pop_size 5 \
    --expected_kt 0.34 \
    --save_dir work_dirs/autoprox_search

conda run -n ERT-NAS python search_autoprox_with_er.py \
    --gt_path ./gt_results/gt_autoformer_2.pth \
    --refer_cfg ./configs/auto/autoformer/autoformer-ti-subnet_flowers_kd.yaml \
    --ds flowers \
    --acc_type kd \
    --num_generations 3 \
    --pop_size 5 \
    --expected_kt 0.70 \
    --save_dir work_dirs/autoprox_search

echo "Auto-Prox search completed!"
