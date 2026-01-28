#!/usr/bin/env bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ERT-NAS

# 定义数组
gt_paths=(
    # './gt_results/gt_autoformer.pth'
    './gt_results/gt_autoformer_2.pth'
)

suffixs=(
    # 'base'
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
for ((i=0; i<${#gt_paths[@]}; i++)); do
    for ((j=0; j<${#datasets[@]}; j++)); do
        # 检查文件是否存在
        if [[ -f "${gt_paths[$i]}" ]]; then
            echo "Running test for gt_paths[$i] = ${gt_paths[$i]}, datasets[$j] = ${datasets[$j]}, suffixs[$i] = ${suffixs[$i]}"
            conda run -n ERT-NAS python test_zc_rank.py --gt_path "${gt_paths[$i]}" --other_zc er --refer_cfg "./configs/auto/autoformer/autoformer-ti-subnet_${_datasets[$j]}_${suffixs[$i]}.yaml" --ds ${datasets[$j]} --acc_type ${suffixs[$i]}
            echo "Test completed successfully!"
        else
            echo "Warning: Ground truth file ${gt_paths[$i]} does not exist. Skipping..."
        fi
    done
done

echo "All tests have been completed!"
# python test_zc_rank.py --gt_path gt_results/gt_autoformer.pth --other_zc er --refer_cfg "./configs/auto/autoformer/autoformer-ti-subnet_chaoyang_base.yaml" --ds chaoyang