import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

sys.path.append('.')
sys.path.append('..')

import torch

from utils.misc import set_random_seed
from utils.logger import write_to_record_file
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB
from r2r.data_utils import construct_instrs
from r2r.env import R2RNavBatch
from r2r.parser import parse_args
from r2r.agent import GMapNavAgent

def build_dataset(args, rank=0, test_file_path=None):
    """
    Build the multi-scene dataset
    
    Args:
        args: arguments
        rank: process rank
        test_file_path: test file path; if None, use the default path from args
    """
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    dataset_class = R2RNavBatch
    
    # Build the training environment (used for model initialization)
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], 
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
        is_test=True
    )
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir,
        batch_size=1,
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train', 
    )
    
    # Read test data directly from the specified JSON file
    if test_file_path is None:
        # Choose the default file based on the mode
        if args.mode == 'generate_cache':
            test_file_path = getattr(args, 'cache_generation_file', None)
        elif args.mode == 'test_with_cache':
            test_file_path = getattr(args, 'cache_testing_file', None)
    
    if test_file_path is None:
        raise ValueError("Test file path not specified")
    
    print(f"Loading test data from: {test_file_path}")
    
    # Load test data
    test_instr_data = construct_instrs(
        args.new_anno_dir, args.dataset, [test_file_path], 
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
        is_test=True
    )
    
    # Group data by scan
    scan_data = defaultdict(list)
    for item in test_instr_data:
        scan_id = item['scan']
        scan_data[scan_id].append(item)
    
    print(f"Found {len(scan_data)} scans in test data:")
    for scan_id, data in scan_data.items():
        print(f"  - {scan_id}: {len(data)} instructions")
    
    # Create a separate env for each scan
    test_envs = {}
    for scan_id, scan_instructions in scan_data.items():
        env_name = f"Test_Residential_Resconstruction:{scan_id}"
        
        test_env = dataset_class(
            feat_db, scan_instructions, args.connectivity_dir, 
            batch_size=1,
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), 
            name=env_name,
        )
        test_envs[env_name] = test_env
    
    print(f"Built {len(test_envs)} test environments")
    return train_env, test_envs

def generate_multi_scene_cache(args, train_env, test_envs, rank=-1):
    """Generate the multi-scene cache"""
    print("=== Generating Multi-Scene Cache ===")
    
    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)
    
    if args.resume_file is not None:
        print("Loading model from %s" % args.resume_file)
        agent.load(args.resume_file)
    
    # Generate the cache file path
    cache_file = getattr(args, 'cache_file', None)
    if cache_file is None:
        cache_file = os.path.join(args.log_dir, 'multi_scene_cache.pkl.gz')
    
    print(f"Will save cache to: {cache_file}")
    
    # Generate the multi-scene cache
    agent.save_multi_scene_cache(cache_file, test_envs)
    
    print("Multi-scene cache generation completed!")
    return cache_file

def test_with_multi_scene_cache(args, train_env, test_envs, rank=-1):
    """Test using the multi-scene cache"""
    default_gpu = is_default_gpu(args)
    
    print("=== Testing with Multi-Scene Cache ===")
    
    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)
    
    if args.resume_file is not None:
        print("Loading model from %s" % args.resume_file)
        agent.load(args.resume_file)
    
    # Load the multi-scene cache
    if hasattr(args, 'multi_scene_cache_file') and args.multi_scene_cache_file:
        print(f"Loading multi-scene cache from: {args.multi_scene_cache_file}")
        cache_loaded = agent.load_multi_scene_cache(args.multi_scene_cache_file)
        if not cache_loaded:
            print("Failed to load multi-scene cache, proceeding without cache")
    else:
        print("No cache file specified, will compute features normally")
    
    if default_gpu:
        record_file = os.path.join(args.log_dir, 'test_results.txt')
        write_to_record_file("Testing with multi-scene cache\n", record_file)
    
    # Test all environments
    all_results = {}
    total_time = 0
    total_instructions = 0
    
    print(f"\nTesting {len(test_envs)} environments...")
    
    for env_name, env in test_envs.items():
        scan_id = env_name.split(":")[-1] if ":" in env_name else env_name
        print(f"\nTesting scene: {scan_id} ({env_name})")
        
        agent.env = env
        
        start_time = time.time()
        agent.test(use_dropout=False, feedback='argmax', iters=None)
        test_time = time.time() - start_time
        total_time += test_time
        
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))
        total_instructions += len(preds)
        
        if default_gpu:
            score_summary, _ = env.eval_metrics(preds)
            all_results[env_name] = {
                'scan_id': scan_id,
                'metrics': score_summary,
                'test_time': test_time,
                'num_instructions': len(preds)
            }
            
            result_str = f"Scene {scan_id}: "
            for metric, val in score_summary.items():
                result_str += f"{metric}: {val:.4f}, "
            result_str += f"time: {test_time:.2f}s, instructions: {len(preds)}"
            
            print(result_str)
            write_to_record_file(result_str + "\n", record_file)
    
    if default_gpu:
        # Compute overall performance
        overall_metrics = defaultdict(list)
        for result in all_results.values():
            for metric, val in result['metrics'].items():
                overall_metrics[metric].append(val)
        
        overall_scores = {}
        for metric, values in overall_metrics.items():
            overall_scores[metric] = np.mean(values)
        
        print(f"\n=== Overall Results ===")
        print(f"Total scenes tested: {len(all_results)}")
        print(f"Total instructions: {total_instructions}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per scene: {total_time/len(all_results):.2f}s")
        print(f"Average time per instruction: {total_time/max(total_instructions,1):.2f}s")
        
        print(f"\nOverall Performance:")
        for metric, val in overall_scores.items():
            print(f"  {metric}: {val:.4f}")
        
        # Save detailed results
        final_results = {
            'overall_metrics': overall_scores,
            'scene_results': all_results,
            'summary': {
                'total_scenes': len(all_results),
                'total_instructions': total_instructions,
                'total_time': total_time,
                'avg_time_per_scene': total_time/len(all_results),
                'avg_time_per_instruction': total_time/max(total_instructions,1)
            }
        }
        
        results_file = os.path.join(args.log_dir, 'multi_scene_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Write overall results to the log
        overall_str = "\n=== Overall Performance ===\n"
        for metric, val in overall_scores.items():
            overall_str += f"{metric}: {val:.4f}\n"
        write_to_record_file(overall_str, record_file)

def main():
    try:
        # Use the existing parser and add new arguments
        args = parse_args()
        
        # Add arguments related to the multi-scene cache
        import argparse
        parser = argparse.ArgumentParser(parents=[argparse.ArgumentParser()], add_help=False)
        parser.add_argument('--mode', choices=['generate_cache', 'test_with_cache'], 
                           help='Operation mode for multi-scene cache')
        parser.add_argument('--multi_scene_cache_file', type=str,
                           help='Path to multi-scene cache file')
        parser.add_argument('--cache_file', type=str,
                           help='Path to save cache file when generating')
        parser.add_argument('--cache_generation_file', type=str,
                           help='JSON file with 500 instructions per scan for cache generation')
        parser.add_argument('--cache_testing_file', type=str,
                           help='JSON file with 100 instructions per scan for cache testing')
        
        # Parse the newly added arguments
        extra_args, _ = parser.parse_known_args()
        
        # Merge arguments
        for key, value in vars(extra_args).items():
            if value is not None:
                setattr(args, key, value)
        
        # Check required arguments
        if not hasattr(args, 'mode') or args.mode is None:
            raise ValueError("--mode is required. Choose from 'generate_cache' or 'test_with_cache'")
        
        # Initialize distributed training
        if args.world_size > 1:
            rank = init_distributed(args)
            torch.cuda.set_device(args.local_rank)
        else:
            rank = 0

        set_random_seed(args.seed + rank)
        
        # Choose the test file based on the mode
        test_file_path = None
        if args.mode == 'generate_cache':
            if hasattr(args, 'cache_generation_file') and args.cache_generation_file:
                test_file_path = args.cache_generation_file
            else:
                raise ValueError("--cache_generation_file is required for generate_cache mode")
        elif args.mode == 'test_with_cache':
            if hasattr(args, 'cache_testing_file') and args.cache_testing_file:
                test_file_path = args.cache_testing_file
            else:
                raise ValueError("--cache_testing_file is required for test_with_cache mode")
        
        # Build the dataset
        print("Building multi-scene dataset...")
        train_env, test_envs = build_dataset(args, rank=rank, test_file_path=test_file_path)
        
        # Run the corresponding operation based on the mode
        if args.mode == 'generate_cache':
            print("Mode: Generate multi-scene cache")
            generate_multi_scene_cache(args, train_env, test_envs, rank=rank)
            
        elif args.mode == 'test_with_cache':
            print("Mode: Test with multi-scene cache")
            test_with_multi_scene_cache(args, train_env, test_envs, rank=rank)
        
        print("Operation completed successfully!")
        
    except Exception as e:
        print(f"Error in main(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()