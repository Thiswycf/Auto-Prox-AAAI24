import argparse
import csv
import os
import torch
import torch.nn.functional as F
import pycls.core.config as config
from pycls.core.config import cfg
import pycls.core.logging as logging
import pycls.datasets.loader as data_loader
from pycls.models.build import MODEL
from pycls.predictor.pruners.predictive import find_measures

logger = logging.get_logger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def main():
    parser = argparse.ArgumentParser(description='Calculate zero-cost proxy score',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--save_dir', type=str, default='work_dirs/proxy_zc')
    parser.add_argument('--csv', action='store_true', help='Save score to CSV file')
    parser.add_argument('--csv_dir', type=str, default='work_dirs/proxy_zc')
    parser.add_argument('--refer_cfg', type=str, default='./configs/auto/autoformer/autoformer-ti-subnet_c100_base.yaml',
                        help='Reference configuration file')
    parser.add_argument('--other_zc', type=str, default=None,
                        help='Zero-cost proxy method: size, epe_nas, grasp, snip, ntk, fisher, synflow, dss, er')
    parser.add_argument('--ds', type=str, default='cifar100',
                        help='Dataset: flowers, cifar100, chaoyang')
    
    args = parser.parse_args()
    
    # Load configuration
    config.load_cfg(args.refer_cfg)
    config.assert_cfg()
    
    # Set proxy dataset
    cfg.PROXY_DATASET = args.ds
    
    # Setup logging
    logging.setup_logging()
    
    # Create save directory if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    if args.csv and not os.path.exists(args.csv_dir):
        os.makedirs(args.csv_dir, exist_ok=True)
    
    # Build model
    if cfg.MODEL.TYPE == 'AutoFormerSub':
        # For AutoFormerSub, use the subnet configuration
        model = MODEL.get(cfg.MODEL.TYPE)()
    elif cfg.MODEL.TYPE == 'PIT':
        # For PIT, use the subnet configuration
        model = MODEL.get(cfg.MODEL.TYPE)()
    else:
        # For other models, use default configuration
        model = MODEL.get(cfg.MODEL.TYPE)()
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
    
    # Construct proxy data loader
    proxy_loader = data_loader.construct_proxy_loader()
    
    # Calculate zero-cost proxy score
    if args.other_zc:
        logger.info(f'Calculating {args.other_zc} score...')
        
        # Set dataload info based on proxy method
        if args.other_zc == 'grasp':
            dataload_info = ['grasp', 1, cfg.MODEL.NUM_CLASSES]
        else:
            dataload_info = ['random', 1, cfg.MODEL.NUM_CLASSES]
        
        # Calculate score
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        score = find_measures(model, proxy_loader,
                              dataload_info=dataload_info,
                              device=device,
                              loss_fn=F.cross_entropy,
                              measure_names=[args.other_zc])
        
        logger.info(f'{args.other_zc} score: {score}')
        
        # Save score to CSV file
        if args.csv:
            csv_filename = f'{cfg.MODEL.TYPE}_{args.other_zc}_{cfg.PROXY_DATASET}.csv'
            csv_path = os.path.join(args.csv_dir, csv_filename)
            
            # Check if file exists, if not create it with header
            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([cfg.PROXY_DATASET])
            
            # Append score to CSV file
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([score])
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
