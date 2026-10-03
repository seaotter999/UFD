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
import wandb

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB
from r2r.data_utils import construct_instrs, get_scans
from r2r.env import R2RNavBatch
from r2r.parser import parse_args

from r2r.agent import GMapNavAgent

def build_dataset(args, rank=0, is_test=False):
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    dataset_class = R2RNavBatch
    train_dataset_splits = [os.path.join(args.new_anno_dir, 'merged_train_dataset.json')]


    my_train_instr_data = construct_instrs(
        args.new_anno_dir, args.dataset, train_dataset_splits, 
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
        is_test=is_test
    )
    custome_train_env = dataset_class(
        feat_db, my_train_instr_data, args.connectivity_dir,
        batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train', 
    )

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, 
            batch_size=args.batch_size, angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
        )
    else:
        aug_env = None

    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], 
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
        is_test=is_test
    )
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir,
        batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train', 
    )
    val_env_names = ['Validation_Residential_Reconstruction']

    dataset_info = {
        'train_dataset': [os.path.basename(split) for split in train_dataset_splits],
        'val_dataset': val_env_names,
        'aug_dataset': os.path.basename(args.aug) if args.aug else None
    }

    env2scans_val = get_scans(args.anno_dir, val_env_names)
    val_env_names = []
    for env_name, scans in env2scans_val.items():
        for scan in scans:
            val_env_names.append(f"{env_name}:{scan}")

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=1,
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
        )   # evaluation using all objects
        val_envs[split] = val_env

    return train_env, custome_train_env, aug_env, val_envs, env2scans_val, dataset_info

def evaluate_by_scan(env, preds):
    """
    Evaluate model performance grouped by scan
    Returns: (scan_scores, overall_scores)
    """
    # Group predictions by scan
    scan_preds = defaultdict(list)
    scan_data = defaultdict(list)
    
    for i, pred in enumerate(preds):
        scan_id = pred.get('scan', env.data[i].get('scan'))  # Get scan_id from the prediction or the original data
        scan_preds[scan_id].append(pred)
        scan_data[scan_id].append(env.data[i])
    
    # Compute performance for each scan
    scan_scores = {}
    all_scan_metrics = defaultdict(list)
    
    for scan_id in scan_preds:
        # Create a temporary environment for evaluating the current scan
        temp_data = scan_data[scan_id]
        temp_preds = scan_preds[scan_id]
        
        # Compute metrics for the current scan
        score_summary, _ = env.eval_metrics(temp_preds)
        scan_scores[scan_id] = score_summary
        
        # Collect metrics from all scans to compute the average
        for metric, value in score_summary.items():
            all_scan_metrics[metric].append(value)
    
    # Compute average performance across all scans
    overall_scores = {}
    for metric, values in all_scan_metrics.items():
        overall_scores[metric] = np.mean(values)
    
    return scan_scores, overall_scores

def train(args, train_env, custome_train_env, aug_env, val_envs, env2scans_val, dataset_info, rank=-1,):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'dataset_info.json'), 'w') as outf:
            json.dump(dataset_info, outf, indent=4)

        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
        
        # Log dataset information
        dataset_log = f"\nUsed Datasets:\n"
        dataset_log += f"  - Training: {', '.join(dataset_info['train_dataset'])}\n"
        dataset_log += f"  - Validation: {', '.join(dataset_info['val_dataset'])}\n"
        if dataset_info['aug_dataset']:
            dataset_log += f"  - Augmentation: {dataset_info['aug_dataset']}\n"
        dataset_log += "\n"
        
        write_to_record_file(dataset_log, record_file)

    if args.wandb_project:
        # Add an identifier for training on the new dataset
        run_name = args.wandb_run_name if hasattr(args, 'wandb_run_name') else None
        if args.use_new_dataset and run_name:
            run_name = f"{run_name}_new_dataset"
        wandb_config = vars(args).copy()
        wandb_config.update(dataset_info)
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=wandb_config
        )

    agent_class = GMapNavAgent
    listener = agent_class(args, train_env, rank=rank)

    # resume file
    if args.resume_file is not None:
        start_iter = listener.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )

    best_val_overall = {
        "spl": 0., 
        "sr": 0., 
        "state": "",
        "all_scan_scores": {}  # Record per-scan performance at the time of the best result
    }
    
    # Best model for each scan
    best_val_by_scan = {}  # {scan_name: {"spl": 0., "sr": 0., "state": ""}}
    
    # Extract all scan names from val_envs
    val_scan_names = []
    for env_name in val_envs.keys():
        if ":" in env_name:
            scan_name = env_name.split(":")[-1]
            val_scan_names.append(scan_name)
            best_val_by_scan[scan_name] = {"spl": 0., "sr": 0., "state": ""}
    
    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )
    best_val = {k.split(":")[0]: {"spl": 0., "sr": 0., "state":""} for k in env2scans_val.keys()}
    start_iter = 0
    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listener.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listener.env = train_env
            listener.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 3))
            for jdx in range(interval // 3):
                # Train with GT data
                listener.env = train_env
                listener.train(1, feedback=args.feedback)

                listener.env = custome_train_env
                listener.train(1, feedback=args.feedback)

                # Train with Augmented data
                listener.env = aug_env
                listener.train(1, feedback=args.feedback)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listener.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listener.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listener.logs['critic_loss']) / total
            policy_loss = sum(listener.logs['policy_loss']) / total
            RL_loss = sum(listener.logs['RL_loss']) / max(len(listener.logs['RL_loss']), 1)
            IL_loss = sum(listener.logs['IL_loss']) / max(len(listener.logs['IL_loss']), 1)
            entropy = sum(listener.logs['entropy']) / total
            if args.wandb_project:
                log_data = {
                    "loss/critic": critic_loss,
                    "policy_entropy": entropy,
                    "loss/RL_loss": RL_loss,
                    "loss/IL_loss": IL_loss,
                    "total_actions": total,
                    "max_length": length,
                    "iter": iter
                }
                if args.use_new_dataset:
                    # Add a prefix to metrics for the new dataset
                    log_data = {f"new_dataset/{k}": v for k, v in log_data.items()}
                wandb.log(log_data)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, RL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, RL_loss, policy_loss, critic_loss),
                record_file
            )

        # Run validation
        loss_str = "iter {}".format(iter)
        
        # Collect evaluation results for all scans
        all_scan_scores = {}
        all_scan_metrics = defaultdict(list)
        
        for env_name, env in val_envs.items():
            if ":" not in env_name:
                continue  # Skip environments that are not scan-specific
                
            scan_name = env_name.split(":")[-1]
            listener.env = env
            
            # Get validation distance from goal under test evaluation conditions
            listener.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listener.get_results()
            preds = merge_dist_results(all_gather(preds))
            
            if default_gpu:
                # Evaluate performance on the current scan
                score_summary, _ = env.eval_metrics(preds)
                all_scan_scores[scan_name] = score_summary
                
                # Collect for computing overall performance
                for metric, val in score_summary.items():
                    all_scan_metrics[metric].append(val)
                
                # Check and save the best model for the current scan
                if score_summary['spl'] >= best_val_by_scan[scan_name]['spl']:
                    best_val_by_scan[scan_name]['spl'] = score_summary['spl']
                    best_val_by_scan[scan_name]['sr'] = score_summary['sr']
                    best_val_by_scan[scan_name]['state'] = 'Iter %d spl: %.2f sr: %.2f' % (
                        iter, score_summary['spl'], score_summary['sr'])
                    listener.save(iter, os.path.join(args.ckpt_dir,
                                 f"best_scan_{scan_name}_iter_{iter}"))
        
        # Compute average performance across all scans (overall performance)
        if default_gpu and all_scan_metrics:
            overall_scores = {}
            for metric, values in all_scan_metrics.items():
                overall_scores[metric] = np.mean(values)
            
            # Append to loss_str for printing
            loss_str += ", overall"
            for metric, val in overall_scores.items():
                loss_str += ', %s: %.2f' % (metric, val)
            
            # Check and save the global best model
            if overall_scores['spl'] >= best_val_overall['spl']:
                best_val_overall['spl'] = overall_scores['spl']
                best_val_overall['sr'] = overall_scores['sr']
                best_val_overall['state'] = 'Iter %d overall spl: %.2f sr: %.2f' % (
                    iter, overall_scores['spl'], overall_scores['sr'])
                best_val_overall['all_scan_scores'] = all_scan_scores.copy()
                listener.save(iter, os.path.join(args.ckpt_dir, 
                             f"best_overall_iter_{iter}"))
            
            # Log detailed per-scan performance
            write_to_record_file(f"\nValidation scan-wise performance at iter {iter}:", record_file)
            for scan_name, scan_score in all_scan_scores.items():
                scan_log = f"  {scan_name}: "
                for metric, val in scan_score.items():
                    scan_log += f"{metric}: {val:.2f}, "
                write_to_record_file(scan_log.rstrip(", "), record_file)
            
            # Log to wandb
            if args.wandb_project:
                val_log_data = {}
                
                # Overall performance
                for metric, val in overall_scores.items():
                    key = f'val/overall/{metric}'
                    if args.use_new_dataset:
                        key = f'new_dataset/{key}'
                    val_log_data[key] = val
                
                # Per-scan performance
                for scan_name, scan_score in all_scan_scores.items():
                    for metric, val in scan_score.items():
                        key = f'val/scan_{scan_name}/{metric}'
                        if args.use_new_dataset:
                            key = f'new_dataset/{key}'
                        val_log_data[key] = val
                
                val_log_data['iter'] = iter
                wandb.log(val_log_data)
            
            # Print current progress and results
            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), 
                                      iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            
            # Print best results
            write_to_record_file("\n=== BEST RESULTS SO FAR ===", record_file)
            write_to_record_file(f"Best Overall: {best_val_overall['state']}", record_file)
            write_to_record_file("\nBest for each scan:", record_file)
            for scan_name, best_info in best_val_by_scan.items():
                if best_info['state']:  # Only print scans that have results
                    write_to_record_file(f"  {scan_name}: {best_info['state']}", record_file)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    results = {}
    for env_name, env in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            # Evaluate performance by scan
            scan_scores, overall_scores = evaluate_by_scan(env, preds)
            
            # Save results
            results[env_name] = {
                'overall': overall_scores,
                'by_scan': scan_scores
            }
            
    if default_gpu:
        with open(os.path.join(args.log_dir, f'best_valid_{args.resume_file.split("/")[-1]}.json'), 'w') as f:
            json.dump(results, f, indent=4)

def check_data_order(env, label=""):
    """Check the current data order"""
    print(f"\n📊 Data Order Check - {label}")
    print("-" * 40)
    
    # Check the instr_id of the first 10 items
    first_10_ids = [item['instr_id'] for item in env.data[:10]]
    print(f"First 10 instr_ids: {first_10_ids}")
    
    # Check the total data count
    print(f"Total data count: {len(env.data)}")
    
    # Check the current index
    print(f"Current index (ix): {env.ix}")
    
    # If path info is available, check it too
    if 'path_id' in env.data[0]:
        first_10_path_ids = [item['path_id'] for item in env.data[:10]]
        print(f"First 10 path_ids: {first_10_path_ids}")
    
    # Check the scan distribution
    scan_counts = defaultdict(int)
    for item in env.data[:100]:  # Check the scan distribution of the first 100 items
        scan_counts[item.get('scan', 'unknown')] += 1
    print(f"Scan distribution (first 100): {dict(scan_counts)}")
    
    return first_10_ids

def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, custome_train_env, aug_env, val_envs, env2scans_val, dataset_info = build_dataset(args, rank=rank, is_test=args.test)

    if not args.test:
        train(args, train_env, custome_train_env, aug_env, val_envs, env2scans_val, dataset_info, rank=rank)
    else:
        valid(args, train_env, val_envs, rank=rank)
            

if __name__ == '__main__':
    main()