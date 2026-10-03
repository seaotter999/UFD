import argparse
import os



def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='r2r', choices=['r2r', 'r4r'])
    parser.add_argument('--output_dir', type=str, default='default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    parser.add_argument('--act_visited_nodes', action='store_true', default=False)
    parser.add_argument('--fusion', choices=['global', 'local', 'avg', 'dynamic'])
    parser.add_argument('--expl_sample', action='store_true', default=False)
    parser.add_argument('--expl_max_ratio', type=float, default=0.6)
    parser.add_argument('--expert_policy', default='spl', choices=['spl', 'ndtw'])

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=100000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--eval_first', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=80)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    
    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default=None, help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)

    parser.add_argument("--features", type=str, default='vitbase')

    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_pano_embedding', action='store_true', default=False)
    parser.add_argument('--fix_local_branch', action='store_true', default=False)

    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_pano_layers', type=int, default=2)
    parser.add_argument('--num_x_layers', type=int, default=4)

    parser.add_argument('--enc_full_graph', default=False, action='store_true')
    parser.add_argument('--graph_sprels', action='store_true', default=False)

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.3)

    # Submision configuration
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_backtrack', action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='rms',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=2048)
    parser.add_argument('--obj_feat_size', type=int, default=0)
    parser.add_argument('--views', type=int, default=36)

    # # A2C
    parser.add_argument("--gamma", default=0.9, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )
    parser.add_argument('--train_alg', 
        choices=['imitation', 'dagger'], 
        default='imitation'
    )

    parser.add_argument('--max_traj_num', type=int, default=500)


    parser.add_argument('--img_ft_file', type=str, default='')
    parser.add_argument('--connectivity_dir', type=str, default='')
    parser.add_argument('--anno_dir', type=str, default='')
    parser.add_argument('--new_anno_dir', type=str, default='')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--use_new_dataset', action='store_true', default=False, help='Use new dataset for training')


    parser.add_argument('--all_scanvp_cands', type=str, default='',)

    parser.add_argument('--pano_inputs_path', type=str, default='',)
    parser.add_argument('--enable_cache', action='store_true', default=False)
    parser.add_argument('--collect_training_data', action='store_true', default=False,
                        help='Enable collection of training data from custome_train_env')
    parser.add_argument('--collect_data_dir',  default='')
    
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
    

    # parser = add_fstta_args(parser)
    parser = add_atena_args(parser)
    # parser = add_tent_args(parser)

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):

    ROOTDIR = args.root_dir
    # Setup input paths
    # args.img_ft_file = os.path.join('../datasets', 'R2R', 'features', 'clip_vit-b16_mp3d_hm3d_gibson.hdf5')
    # args.connectivity_dir = os.path.join('../datasets', 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')
    # args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations', 'ESA_Dataset')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args

def add_fstta_args(parser):
    """
    Add FSTTA-specific arguments to the argument parser.
    
    Usage:
        from parser_fstta import add_fstta_args
        add_fstta_args(parser)
    """
    
    # FSTTA Enable/Disable
    parser.add_argument('--use_fstta', action='store_true', default=False,
                        help='Enable Fast-Slow Test-Time Adaptation')
    
    # FAST Update Parameters
    parser.add_argument('--fstta_lr_fast', type=float, default=6e-4,
                        help='Base learning rate for FAST updates')
    parser.add_argument('--fstta_M', type=int, default=3,
                        help='FAST update interval (number of action steps)')
    parser.add_argument('--fstta_rho', type=float, default=0.9,
                        help='Momentum for historical variance update in FAST')
    parser.add_argument('--fstta_tau', type=float, default=0.5,
                        help='Threshold for dynamic learning rate scaling')
    parser.add_argument('--fstta_lr_scale_min', type=float, default=0.5,
                        help='Minimum learning rate scale factor')
    parser.add_argument('--fstta_lr_scale_max', type=float, default=1.5,
                        help='Maximum learning rate scale factor')
    
    # SLOW Update Parameters
    parser.add_argument('--fstta_lr_slow', type=float, default=1e-3,
                        help='Learning rate for SLOW updates')
    parser.add_argument('--fstta_N', type=int, default=4,
                        help='SLOW update interval (number of test samples)')
    parser.add_argument('--fstta_q', type=float, default=0.1,
                        help='Exponential weight factor for reference direction')
    
    return parser

def add_atena_args(parser):
        # ============== ATENA-specific Arguments ==============
    # These are the key hyperparameters for ATENA algorithm
    parser.add_argument('--active_criteria', type=float, default=0.1,
                        help='Uncertainty threshold δ for active query decision. '
                             'If avg_entropy > δ, query human oracle; else use self oracle. '
                             'Paper uses δ ∈ {0.1, 0.2, 0.3}')
    parser.add_argument('--lr_pre', type=float, default=8e-7,
                        help='Learning rate for uncertain episodes (human oracle). '
                             'Higher LR for informative samples.')
    parser.add_argument('--lr_post', type=float, default=1e-7,
                        help='Learning rate for certain episodes (self oracle). '
                             'Lower LR to prevent overconfident updates.')
    parser.add_argument('--atena_lr', type=float, default=1e-6, help='Test-time adaptation learning rate (eta)')
    parser.add_argument('--bin_weight', type=float, default=0.1,
                        help='Weight γ for self-prediction loss in total loss. '
                             'L = L_mix + γ * L_self')
    parser.add_argument('--r_lambda', type=float, default=0.75,
                        help='Mixture weight λ for pseudo-expert distribution. '
                             'q_mix = λ * q_pseudo + (1-λ) * π_θ. '
                             'Paper shows optimal around λ=0.4-0.8')
    return parser



def add_tent_args(parser):
    """Add TENT-specific arguments.

    TENT: Fully Test-Time Adaptation by Entropy Minimization.

    We adapt the VLN policy at test time by minimizing the entropy of the
    action distribution, updating only a small set of normalization affine
    parameters (LayerNorm by default).
    """

    parser.add_argument('--use_tent', action='store_true', default=False,
                        help='Enable TENT test-time adaptation (entropy minimization)')

    parser.add_argument('--tent_lr', type=float, default=6e-5,
                        help='Learning rate for TENT updates')
    parser.add_argument('--tent_steps', type=int, default=1,
                        help='Number of optimizer steps per test-time update call')

    parser.add_argument('--tent_k_ln', type=int, default=4,
                        help='Update only the last-k LayerNorm params (<=0 means all LayerNorm params)')

    parser.add_argument('--tent_disable_dropout', action='store_true', default=False,
                        help='Disable dropout during TENT (recommended for Transformer-based VLN)')

    parser.add_argument('--tent_episodic', action='store_true', default=False,
                        help='Reset adaptation state per episode (trajectory)')

    return parser