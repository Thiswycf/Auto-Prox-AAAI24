"""
Auto-Prox search script that includes er in the search space.
This script searches for optimal ZCP formulas using Auto-Prox algorithm.
"""
import argparse
import random
import os
import pycls.core.config as config
import pycls.core.logging as logging
import pycls.datasets.loader as data_loader
from pycls.core.config import cfg
from autozc.structures import GraphStructure, LinearStructure, TreeStructure
from test_zc_rank import auto_prox_fitness
import torch
import time
import yaml
import copy
import shutil

logger = logging.get_logger(__name__)


def generate_autoprox_candidates(num_candidates=100):
    """
    Generate Auto-Prox candidate formulas.
    Includes er in the search space along with other ZCPs.
    """
    candidates = []
    
    # Define available ZCP identifiers
    available_zcps = ['t2', 't4g', 'er', 'ntk', 'snip', 'fisher', 'dss', 'nwot']
    
    # Define operations
    unary_ops = ['min_max_normalize', 'z_score_normalize', 'frobenius_norm', 
                 'sigmoid', 'element_wise_sqrt', 'element_wise_square', 'log', 'abs']
    binary_ops = ['element_wise_sum', 'element_wise_mul', 'element_wise_sub', 'element_wise_div']
    
    for _ in range(num_candidates):
        # Randomly select 2-3 input ZCPs (including er)
        num_inputs = random.randint(2, 3)
        input_zcps = random.sample(available_zcps, num_inputs)
        
        # Ensure er is included in candidates
        if 'er' not in input_zcps:
            input_zcps[0] = 'er'
        
        # Generate tree structure
        num_ops = random.randint(1, 3)
        selected_ops = random.sample(unary_ops, min(num_ops, len(unary_ops)))
        binary_op = random.choice(binary_ops)
        
        # Create repr_geno string
        input_str = ','.join(input_zcps)
        ops_str = '|'.join(selected_ops)
        repr_geno = f'INPUT:({input_str})TREE:({ops_str})BINARY:({binary_op})'
        
        # Generate op_geno (weights for linear combination)
        op_geno = [[random.uniform(0.1, 2.0) for _ in range(len(input_zcps))]]
        
        candidate = {
            'input_geno': input_zcps,
            'op_geno': op_geno,
            'repr_geno': repr_geno,
            'type': 'tree'  # Can be 'tree', 'linear', or 'graph'
        }
        
        candidates.append(candidate)
    
    return candidates


def evaluate_candidate(cfg, candidate, arch_pop, acc_pop, data_loader, num_classes):
    """Evaluate a candidate Auto-Prox formula."""
    try:
        # Create structure based on type
        if candidate['type'] == 'tree':
            structure = TreeStructure()
        elif candidate['type'] == 'linear':
            structure = LinearStructure()
        else:
            structure = GraphStructure()
        
        # Set genotype
        structure._genotype['input_geno'] = candidate['input_geno']
        structure._genotype['op_geno'] = candidate['op_geno']
        structure._genotype['repr_geno'] = candidate['repr_geno']
        
        # Evaluate fitness
        result = auto_prox_fitness(cfg, data_loader, arch_pop, acc_pop, structure, num_classes)
        
        if result == -1:
            return None
        
        ken, sp, ps = result
        return {
            'kendall': ken,
            'spearman': sp,
            'pearson': ps,
            'fitness': ken  # Use Kendall as fitness
        }
    except Exception as e:
        logger.warning(f"Error evaluating candidate: {e}")
        return None


def search_autoprox(cfg, arch_pop, acc_pop, data_loader, num_classes, num_generations=10, pop_size=20, expected_kt=0.0):
    """Evolutionary search for optimal Auto-Prox formula."""
    logger.info("Starting Auto-Prox search with er in search space...")
    
    # Initial population
    population = generate_autoprox_candidates(pop_size)
    
    best_candidate = None
    best_fitness = float('-inf')
    
    for generation in range(num_generations):
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        logger.info(f"[{time_str}] Generation {generation + 1}/{num_generations}")
        
        # Evaluate all candidates
        fitness_scores = []
        for i, candidate in enumerate(population):
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
            logger.info(f"[{time_str}] Evaluating candidate {i+1}/{len(population)}: {candidate['input_geno']}")
            result = evaluate_candidate(cfg, candidate, arch_pop, acc_pop, data_loader, num_classes)
            
            if result:
                fitness = result['fitness']
                fitness_scores.append((candidate, fitness, result))
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_candidate = candidate.copy()
                    best_candidate['results'] = result
                    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
                    logger.info(f"[{time_str}] New best candidate found! Fitness: {fitness:.4f}, "
                              f"Kendall: {result['kendall']:.4f}, "
                              f"Spearman: {result['spearman']:.4f}, "
                              f"Pearson: {result['pearson']:.4f}")
                    if best_fitness >= expected_kt:
                        logger.info(f"[{time_str}] Best fitness {best_fitness:.4f} >= expected_kt {expected_kt:.4f}")
            else:
                fitness_scores.append((candidate, float('-inf'), None))
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Selection: keep top 50%
        elite_size = max(1, len(fitness_scores) // 2)
        elite = [cand for cand, fit, res in fitness_scores[:elite_size] if res is not None]
        
        # Generate new population
        new_population = elite.copy()
        
        # Mutation: mutate elite candidates
        while len(new_population) < pop_size:
            # Handle empty elite case
            if not elite:
                parent = random.choice(population)
            else:
                parent = random.choice(elite)
            child = copy.deepcopy(parent)
            
            # Mutate input_geno
            if random.random() < 0.3:
                available_zcps = ['t2', 't4g', 'er', 'ntk', 'snip', 'fisher', 'dss', 'nwot']
                child['input_geno'] = random.sample(available_zcps, len(parent['input_geno']))
            
            # Mutate repr_geno
            if random.random() < 0.3:
                unary_ops = ['min_max_normalize', 'z_score_normalize', 'frobenius_norm', 
                           'sigmoid', 'element_wise_sqrt', 'element_wise_square']
                binary_ops = ['element_wise_sum', 'element_wise_mul', 'element_wise_sub']
                
                input_str = ','.join(child['input_geno'])
                ops_str = '|'.join(random.sample(unary_ops, random.randint(1, 3)))
                binary_op = random.choice(binary_ops)
                child['repr_geno'] = f'INPUT:({input_str})TREE:({ops_str})BINARY:({binary_op})'
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    return best_candidate, best_fitness


def parse_args():
    parser = argparse.ArgumentParser(
        description='Auto-Prox search with er in search space',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--gt_path', type=str, required=True, help='Ground truth results path')
    parser.add_argument('--refer_cfg', type=str, required=True, help='Reference config file')
    parser.add_argument('--ds', type=str, default='cifar100', help='Dataset: flowers, cifar100, chaoyang')
    parser.add_argument('--acc_type', type=str, default='kd', help='Accuracy type: base, kd')
    parser.add_argument('--num_generations', type=int, default=5, help='Number of generations')
    parser.add_argument('--pop_size', type=int, default=10, help='Population size')
    parser.add_argument('--expected_kt', type=float, default=0.0, help='Expected Kendall\'s tau')
    parser.add_argument('--save_dir', type=str, default='work_dirs/autoprox_search', help='Save directory')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config.load_cfg(args.refer_cfg)
    config.assert_cfg()
    cfg.PROXY_DATASET = args.ds
    
    logging.setup_logging()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    
    save_dir = os.path.join(args.save_dir, f"{cfg.MODEL.TYPE}_{args.ds}_{args.acc_type}_{time_str}")
    os.makedirs(save_dir, exist_ok=True)
    
    log_file = os.path.join(save_dir, f'autoprox_search_{time_str}.txt')
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info("=" * 50)
    logger.info("Auto-Prox Search with er in Search Space")
    logger.info("=" * 50)
    logger.info(f"Model: {cfg.MODEL.TYPE}")
    logger.info(f"Dataset: {args.ds}")
    logger.info(f"Accuracy type: {args.acc_type}")
    logger.info(f"Generations: {args.num_generations}")
    logger.info(f"Population size: {args.pop_size}")
    
    # Load ground truth
    from test_zc_rank import obtain_gt
    gt_results = torch.load(args.gt_path)
    arch_pop, acc_pop = obtain_gt(gt_results, args.ds, args.acc_type)
    
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    logger.info(f"[{time_str}] Loaded {len(arch_pop)} architectures")
    
    # Create data loader
    data_loader = data_loader.construct_proxy_loader()
    
    # Run search
    t1 = time.time()
    best_candidate, best_fitness = search_autoprox(
        cfg, arch_pop, acc_pop, data_loader, 
        cfg.MODEL.NUM_CLASSES, 
        num_generations=args.num_generations,
        pop_size=args.pop_size,
        expected_kt=args.expected_kt
    )
    t2 = time.time()
    
    if best_fitness > args.expected_kt:
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        logger.info(f"[{time_str}] Search Completed!")
        logger.info("=" * 50)
        logger.info(f"Best candidate: {best_candidate}")
        logger.info(f"Best fitness (Kendall): {best_fitness:.4f}")
        if best_candidate and 'results' in best_candidate:
            logger.info(f"Kendall: {best_candidate['results']['kendall']:.4f}")
            logger.info(f"Spearman: {best_candidate['results']['spearman']:.4f}")
            logger.info(f"Pearson: {best_candidate['results']['pearson']:.4f}")
        logger.info(f"Time cost: {(t2 - t1) / 3600:.2f} hours")

        # Save best candidate
        best_candidate_file = os.path.join(save_dir, 'best_candidate.yaml')
        with open(best_candidate_file, 'w') as f:
            yaml.dump(best_candidate, f, default_flow_style=False)

        logger.info(f"Best candidate saved to: {best_candidate_file}")
    else:
        shutil.rmtree(save_dir)
