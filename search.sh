#!/usr/bin/env bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="3"

set -e

python search_model_shell.py --trial_num 1000 --gpu_idx 0 --refer_cfg configs/auto/autoformer/autoformer-ti-subnet_c100_base.yaml --other_zc er