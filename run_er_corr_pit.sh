#!/usr/bin/env bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="3"

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ERT-NAS

# 定义数组
suffixs=(
    'base'
    'kd'
)

datasets=(
    'cifar100'
    'flowers'
    'chaoyang'
)

_datasets=(
    'c100'
    'flowers'
    'chaoyang'
)

# 遍历所有组合
for ((i=0; i<${#suffixs[@]}; i++)); do
    for ((j=0; j<${#datasets[@]}; j++)); do
        # 检查文件是否存在
        if [[ -f "./gt_results/gt_pit.pth" ]]; then
            echo "Running test for gt_paths[$i] = ./gt_results/gt_pit.pth, datasets[$j] = ${datasets[$j]}, suffixs = ${suffixs[$i]}"
            conda run -n ERT-NAS python test_zc_rank.py --gt_path ./gt_results/gt_pit.pth --other_zc er --refer_cfg "configs/auto/pit/pit-ti_${_datasets[$j]}_${suffixs[$i]}.yaml" --ds ${datasets[$j]} --acc_type ${suffixs[$i]}
            echo "Test completed successfully!"
        else
            echo "Warning: Ground truth file ./gt_results/gt_pit.pth does not exist. Skipping..."
        fi
    done
done

echo "All tests have been completed!"

# python test_zc_rank.py --gt_path ./gt_results/gt_pit.pth --other_zc er --refer_cfg configs/auto/pit/pit-ti_c100_kd.yaml --ds cifar100