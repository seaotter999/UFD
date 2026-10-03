import os
import json
import numpy as np
import networkx as nx
import types
from collections import defaultdict
from tqdm import tqdm

def load_topology_from_cache(cache_file):
    """Load topology graphs from a cache file"""
    import pickle
    import gzip
    import networkx as nx
    
    print(f"Loading topology from cache: {cache_file}")
    
    try:
        if cache_file.endswith('.gz'):
            with gzip.open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
        else:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return {}
    
    graphs = {}
    
    # Handle the multi-scene cache format
    if 'scene_gmaps' in cache_data:
        for scan_id, gmaps in cache_data['scene_gmaps'].items():
            G = nx.Graph()
            
            for gmap in gmaps:
                # Add nodes
                for vp, pos in gmap.node_positions.items():
                    if vp not in G:
                        G.add_node(vp, position=pos)
                
                # Add edges
                for vp1, distances in gmap.graph._dis.items():
                    for vp2, dist in distances.items():
                        if vp1 < vp2 and not G.has_edge(vp1, vp2):
                            G.add_edge(vp1, vp2, weight=dist)
            
            graphs[scan_id] = G
            print(f"  Scene {scan_id}: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Handle the single-scene cache format
    elif 'gmaps' in cache_data:
        gmaps = cache_data['gmaps']
        # Get scan_id from the first gmap
        if gmaps and hasattr(gmaps[0], 'cached_raw_inputs'):
            first_vp = list(gmaps[0].cached_raw_inputs.keys())[0]
            scan_id = gmaps[0].cached_raw_inputs[first_vp]['obs_data']['scan']
            
            G = nx.Graph()
            for gmap in gmaps:
                for vp, pos in gmap.node_positions.items():
                    if vp not in G:
                        G.add_node(vp, position=pos)
                for vp1, distances in gmap.graph._dis.items():
                    for vp2, dist in distances.items():
                        if vp1 < vp2 and not G.has_edge(vp1, vp2):
                            G.add_edge(vp1, vp2, weight=dist)
            
            graphs[scan_id] = G
    
    print(f"Loaded {len(graphs)} scene topologies from cache")
    return graphs


def reconstruct_paths_from_cache(cache_file, feedback_data, output_dir):
    """Reconstruct paths using the topology in the cache (no need to re-run navigation)"""
    import networkx as nx
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load topology from the cache
    graphs = load_topology_from_cache(cache_file)
    
    if not graphs:
        print("Failed to load topology from cache")
        return None, None
    
    def heuristic(G, node1, node2):
        """A* heuristic function"""
        pos1 = G.nodes[node1].get('position', (0, 0, 0))
        pos2 = G.nodes[node2].get('position', (0, 0, 0))
        return np.sqrt(sum((p1-p2)**2 for p1, p2 in zip(pos1, pos2)))
    
    # Reconstruct paths
    reconstructed_data = []
    successful = 0
    
    for item in feedback_data:
        scan = item['scan']
        start = item['start']
        target = item['target']
        
        result = item.copy()
        
        if scan not in graphs:
            print(f"Warning: No topology for scan {scan}")
            result['path'] = [start]
            result['path_reconstruction_success'] = False
        elif start not in graphs[scan] or target not in graphs[scan]:
            print(f"Warning: Start or target not in graph for {item.get('path_id', 'unknown')}")
            result['path'] = [start]
            result['path_reconstruction_success'] = False
        else:
            try:
                G = graphs[scan]
                path = nx.astar_path(G, start, target, 
                                    heuristic=lambda a, b: heuristic(G, a, b),
                                    weight='weight')
                
                # Compute path distance
                distance = sum(G.edges[path[i], path[i+1]]['weight'] 
                              for i in range(len(path)-1))
                
                result['path'] = path
                result['distances'] = round(distance, 2)
                result['path_reconstruction_success'] = True
                result['optimal_path_length'] = len(path)
                successful += 1
                
            except nx.NetworkXNoPath:
                result['path'] = [start]
                result['path_reconstruction_success'] = False
        
        reconstructed_data.append(result)
    
    # Save results
    output_file = os.path.join(output_dir, 'cache_reconstructed_paths.json')
    with open(output_file, 'w') as f:
        json.dump(reconstructed_data, f, indent=2)
    
    print(f"\n=== Cache-based Path Reconstruction Complete ===")
    print(f"Total: {len(feedback_data)}")
    print(f"Successful: {successful}")
    print(f"Success rate: {successful/len(feedback_data)*100:.1f}%")
    print(f"Output: {output_file}")
    
    return output_file, reconstructed_data

class RealTimePathReconstructionCollector:
    """Real-time path reconstruction collector - builds NetworkX graphs during navigation"""
    def __init__(self):
        self.graphs = {}  # scan_id -> NetworkX Graph
        self.navigation_results = []  # Stores navigation results
        self.episode_count = 0
        
        # Data structures built in real time
        self.scan_viewpoints = defaultdict(set)
        self.scan_edges = defaultdict(set)
        self.scan_stats = defaultdict(lambda: {'episodes': 0, 'observations': 0})
        
    def record_user_feedback(self, scan_id, start_vp, target_vp, path, success, 
                            instructions, path_id, heading, instr_encodings):
        """Record the endpoint from user feedback and related information"""
        self.navigation_results.append({
            'scan': scan_id,
            'path_id': path_id,
            'start': start_vp,
            'target': target_vp,
            'actual_path': path,
            'success': success,
            'instructions': instructions,
            'heading': heading,
            'instr_encodings': instr_encodings
        })
        self.episode_count += 1
    
    def add_viewpoint(self, scan_id, viewpoint_id, position):
        """Add a viewpoint to the graph in real time"""
        if scan_id not in self.graphs:
            self.graphs[scan_id] = nx.Graph()
        
        graph = self.graphs[scan_id]
        
        # Add node (if not already present)
        if viewpoint_id not in graph:
            graph.add_node(viewpoint_id, position=position)
            self.scan_viewpoints[scan_id].add(viewpoint_id)
    
    def add_connection(self, scan_id, vp1, vp2, pos1, pos2, distance=None):
        """Add a viewpoint connection to the graph in real time"""
        if scan_id not in self.graphs:
            self.graphs[scan_id] = nx.Graph()
        
        graph = self.graphs[scan_id]
        
        # Ensure both nodes exist
        self.add_viewpoint(scan_id, vp1, pos1)
        self.add_viewpoint(scan_id, vp2, pos2)
        
        # Create a normalized edge representation
        edge = tuple(sorted([vp1, vp2]))
        
        # Avoid adding duplicate edges
        if edge not in self.scan_edges[scan_id]:
            # Compute distance
            if distance is None:
                distance = np.sqrt(sum((p1-p2)**2 for p1, p2 in zip(pos1, pos2)))
            
            # Add edge
            graph.add_edge(vp1, vp2, weight=distance)
            self.scan_edges[scan_id].add(edge)
    
    def reconstruct_optimal_path(self, scan_id, start_vp, target_vp):
        """Reconstruct the optimal path with the A* algorithm based on the endpoint from user feedback"""
        if scan_id not in self.graphs:
            return None
        
        graph = self.graphs[scan_id]
        if start_vp not in graph or target_vp not in graph:
            return None
        
        def heuristic(node1, node2):
            """Heuristic function: uses Euclidean distance"""
            pos1 = graph.nodes[node1].get('position', (0, 0, 0))
            pos2 = graph.nodes[node2].get('position', (0, 0, 0))
            return np.sqrt(sum((p1-p2)**2 for p1, p2 in zip(pos1, pos2)))
        
        try:
            path = nx.astar_path(graph, start_vp, target_vp, heuristic=heuristic, weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

class RealTimePathReconstructionWrapper:
    """Real-time path reconstruction wrapper"""
    def __init__(self, agent, env, collector):
        self.agent = agent
        self.env = env
        self.collector = collector
        self.num_episodes = 0
        self.max_episodes = 500
        self.collected_scans = set()
        
        # Debug flag
        self.debug = True
        
        # Build data mapping
        self.build_data_mapping()
        
    def build_data_mapping(self):
        """Build a mapping from instr_id to the complete data"""
        self.instr_id_to_data = {}
        if hasattr(self.env, 'data'):
            for data_item in self.env.data:
                instr_id = data_item.get('instr_id')
                if instr_id:
                    self.instr_id_to_data[instr_id] = data_item
            print(f"Built data mapping for {len(self.instr_id_to_data)} instructions")
    
    def apply_hooks(self):
        """Apply hook functions to collect topology data in real time"""
        agent_class_name = self.agent.__class__.__name__
        print(f"Applying real-time data collection hooks to {agent_class_name}...")
        
        success = False
        
        # Hook the test method
        if hasattr(self.agent, 'test'):
            self._apply_test_hook()
            success = True
            print("Applied test method hook (builds topology in real time)")
        
        # Hook the rollout method to collect topology in real time
        if hasattr(self.agent, 'rollout'):
            self._apply_rollout_hook()
            success = True
            print("Applied rollout method hook (builds topology in real time)")
            
        if not success:
            print("No suitable hook point found")
        else:
            print(f"Applied real-time data collection hooks to {agent_class_name}")
    
    def _apply_test_hook(self):
        """Hook the agent.test method"""
        original_test = self.agent.test
        
        def wrapped_test(self_agent, *args, **kwargs):
            print("Running test and collecting topology data in real time...")
            
            # Run the original test method
            result = original_test(*args, **kwargs)
            
            # Collect navigation results after the test finishes
            self._collect_navigation_results_improved(self_agent)
            
            return result
        
        self.agent.test = types.MethodType(wrapped_test, self.agent)
    
    def _apply_rollout_hook(self):
        """Hook the rollout method to collect topology in real time"""
        original_rollout = self.agent.rollout
        
        def wrapped_rollout(self_agent, *args, **kwargs):
            # Collect topology in real time during rollout
            result = original_rollout(*args, **kwargs)
            
            # Get the current observations from the environment and extract topology info
            if hasattr(self_agent, 'env') and hasattr(self_agent.env, '_get_obs'):
                try:
                    obs = self_agent.env._get_obs()
                    self._extract_topology_from_obs(obs)
                except Exception as e:
                    if self.debug:
                        print(f"Failed to get observation data: {e}")
            
            return result
        
        self.agent.rollout = types.MethodType(wrapped_rollout, self.agent)
    
    def _extract_topology_from_obs(self, obs):
        """Extract topology info from observation data in real time"""
        if not obs:
            return
        
        for ob in obs:
            scan_id = ob.get('scan')
            if not scan_id:
                continue
            
            viewpoint_id = ob.get('viewpoint')
            position = ob.get('position', (0, 0, 0))
            
            if not viewpoint_id:
                continue
            
            # Add the current viewpoint
            self.collector.add_viewpoint(scan_id, viewpoint_id, position)
            
            # Extract connections from candidates
            candidates = ob.get('candidate', [])
            for cand in candidates:
                # Handle MatterSim.ViewPoint objects
                cand_vp = None
                cand_pos = None
                
                if hasattr(cand, 'viewpointId'):
                    cand_vp = cand.viewpointId
                elif isinstance(cand, dict) and 'viewpointId' in cand:
                    cand_vp = cand['viewpointId']
                
                if hasattr(cand, 'position'):
                    pos = cand.position
                    if hasattr(pos, 'x'):
                        cand_pos = (pos.x, pos.y, pos.z)
                elif isinstance(cand, dict) and 'position' in cand:
                    pos = cand['position']
                    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                        cand_pos = tuple(pos[:3])
                
                if cand_vp and cand_vp != viewpoint_id and cand_pos:
                    # Get distance
                    distance = None
                    if hasattr(cand, 'distance'):
                        distance = cand.distance
                    elif isinstance(cand, dict) and 'distance' in cand:
                        distance = cand['distance']
                    
                    # Add connection
                    self.collector.add_connection(
                        scan_id, viewpoint_id, cand_vp,
                        position, cand_pos, distance
                    )
    
    def _collect_navigation_results_improved(self, agent):
        """Collect navigation results"""
        if not hasattr(agent, 'results') or not agent.results:
            print("Agent has no results data")
            return
        
        print(f"Starting to collect {len(agent.results)} navigation results...")
        collected_count = 0
        
        for instr_id, result in agent.results.items():
            # Get the original data
            data_item = self.instr_id_to_data.get(instr_id)
            
            if not data_item:
                continue
            
            # Get trajectory (actual path)
            trajectory = result.get('trajectory', result.get('path', []))
            if not trajectory:
                continue
            
            # Parse the actual path
            actual_path = []
            for step in trajectory:
                if isinstance(step, list) and step:
                    actual_path.extend(step)
                elif step:
                    actual_path.append(step)
            
            if not actual_path:
                continue
            
            # Extract all required fields
            scan_id = data_item.get('scan')
            path_id = data_item.get('path_id')
            heading = data_item.get('heading', 0.0)
            
            # Handle instructions
            instructions = data_item.get('instructions', [])
            if not instructions:
                instruction = data_item.get('instruction')
                if instruction:
                    instructions = [instruction] if isinstance(instruction, str) else instruction
            
            # Handle instr_encodings
            instr_encodings = data_item.get('instr_encodings', [])
            if not instr_encodings:
                instr_encoding = data_item.get('instr_encoding')
                if instr_encoding:
                    if instr_encoding and isinstance(instr_encoding[0], int):
                        instr_encodings = [instr_encoding]
                    else:
                        instr_encodings = instr_encoding
            
            if not instr_encodings and instructions:
                instr_encodings = [[101, 102]] * len(instructions)
            
            gt_path = data_item.get('path', [])
            
            start_vp = actual_path[0] if actual_path else None
            target_vp = gt_path[-1] if gt_path else actual_path[-1]
            
            if scan_id and start_vp and target_vp:
                success = target_vp in actual_path
                
                self.collector.record_user_feedback(
                    scan_id, start_vp, target_vp, actual_path, success,
                    instructions, path_id, heading, instr_encodings
                )
                collected_count += 1
        
        print(f"Successfully collected {collected_count} navigation results")
        
        # Print statistics of the topology built in real time
        print("\n=== Statistics of Topology Graphs Built in Real Time ===")
        for scan_id, graph in self.collector.graphs.items():
            print(f"Scene {scan_id}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

def reconstruct_paths_from_realtime_feedback(agent, env, output_dir):
    """Reconstruct optimal paths based on endpoints from user feedback (topology built in real time)"""
    print("Starting real-time path reconstruction task...")
    print("Task description: build the topology graph in real time during navigation, then reconstruct optimal paths with the A* algorithm")
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the collector and wrapper
    collector = RealTimePathReconstructionCollector()
    wrapper = RealTimePathReconstructionWrapper(agent, env, collector)
    
    # Apply hooks and run the test
    wrapper.apply_hooks()
    
    print("Running 500 instructions and collecting topology graph info in real time...")
    
    try:
        # Run the test
        agent.test(use_dropout=False, feedback='argmax', iters=None)
        
        # Get prediction results
        preds = agent.get_results()
        
        # Compute navigation performance metrics
        print("=" * 50)
        print("Navigation performance evaluation:")
        score_summary, _ = env.eval_metrics(preds)
        
        metrics_str = "Navigation Performance: "
        for metric, val in score_summary.items():
            metrics_str += f'{metric}: {val:.2f}, '
        print(metrics_str.rstrip(', '))
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during test execution: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Completed {len(collector.navigation_results)} navigation episodes")
    print(f"Collected topology graphs for {len(collector.graphs)} scenes in real time")
    
    # Detailed topology graph statistics
    total_nodes = total_edges = 0
    for scan_id, graph in collector.graphs.items():
        nodes, edges = len(graph.nodes), len(graph.edges)
        total_nodes += nodes
        total_edges += edges
        print(f"Scene {scan_id}: {nodes} nodes, {edges} edges")
    
    print(f"Total: {total_nodes} nodes, {total_edges} edges")
    
    # Generate the first file: user feedback data (start and end points, empty path)
    feedback_data = []
    for result in collector.navigation_results:
        entry = {
            "scan": result['scan'],
            "path_id": result['path_id'],
            "start": result['start'],
            "target": result['target'],
            "path": [],  # Path is empty in the first file
            "heading": result['heading'],
            "distances": 0.0,  # Distance is 0 in the first file
            "instructions": result['instructions'],
            "instr_encodings": result['instr_encodings'],
            "user_feedback": True,
            "navigation_success": result['success']
        }
        feedback_data.append(entry)
    
    # Generate the second file: optimal paths reconstructed with the A* algorithm
    reconstructed_data = []
    successful_reconstructions = 0
    
    for result in collector.navigation_results:
        optimal_path = collector.reconstruct_optimal_path(
            result['scan'], result['start'], result['target']
        )
        
        if optimal_path and len(optimal_path) > 1:
            successful_reconstructions += 1
            graph = collector.graphs[result['scan']]
            
            # Compute the total path distance
            distance = 0.0
            for i in range(len(optimal_path) - 1):
                edge_data = graph.get_edge_data(optimal_path[i], optimal_path[i+1])
                if edge_data:
                    distance += edge_data.get('weight', 0.0)
        else:
            # If the path cannot be reconstructed, include only the start point
            optimal_path = [result['start']]
            distance = 0.0
        
        entry = {
            "scan": result['scan'],
            "path_id": result['path_id'],
            "start": result['start'],
            "target": result['target'],
            "path": optimal_path,
            "heading": result['heading'],
            "distances": round(distance, 2),
            "instructions": result['instructions'],
            "instr_encodings": result['instr_encodings'],
            "navigation_success": result['success'],
            "path_reconstruction_success": len(optimal_path) > 1,
            "optimal_path_length": len(optimal_path)
        }
        reconstructed_data.append(entry)
    
    # Save files
    feedback_data_path = os.path.join(output_dir, 'user_feedback_data.json')
    reconstructed_paths_path = os.path.join(output_dir, 'astar_reconstructed_paths.json')
    
    with open(feedback_data_path, 'w') as f:
        json.dump(feedback_data, f, indent=2)
    
    with open(reconstructed_paths_path, 'w') as f:
        json.dump(reconstructed_data, f, indent=2)
    
    # Print results
    print(f"\n=== Real-time Path Reconstruction Complete ===")
    print(f"User feedback data: {feedback_data_path}")
    print(f"A* reconstructed paths: {reconstructed_paths_path}")
    print(f"Total navigation episodes: {len(feedback_data)}")
    print(f"Successfully reconstructed paths: {successful_reconstructions}")
    print(f"Path reconstruction success rate: {successful_reconstructions/len(feedback_data)*100:.1f}%")
    
    return feedback_data_path, reconstructed_paths_path

def build_dataset(args, rank=0, is_test=True, specified_scan=None):
    """Build the dataset"""
    from utils.data import ImageFeaturesDB
    from r2r.data_utils import construct_instrs
    from r2r.env import R2RNavBatch
    
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    
    new_anno_dir = args.new_anno_dir if hasattr(args, 'new_anno_dir') and args.new_anno_dir else args.anno_dir
    
    # if specified_scan:
    #     dataset_file = f'split_test_500_{specified_scan}.json'
    #     print(f"Using the dataset file for specified scan '{specified_scan}': {dataset_file}")
    # else:
    #     print("Please specify a scan")
    #     return None, None
        
    val_dataset_splits = [new_anno_dir]
    
    if not os.path.exists(val_dataset_splits[0]):
        print(f"Error: dataset file does not exist: {val_dataset_splits[0]}")
        return None, None
    
    print(f"Using dataset file: {val_dataset_splits[0]}")
    
    # Debug: check data contents
    with open(val_dataset_splits[0], 'r') as f:
        raw_data = json.load(f)
        print(f"Dataset contains {len(raw_data)} entries")
    
    test_instr_data = construct_instrs(
        new_anno_dir, args.dataset, val_dataset_splits,
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
        is_test=is_test
    )
    
    test_env = R2RNavBatch(
        feat_db, test_instr_data, args.connectivity_dir,
        batch_size=1,
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size),
        name='test',
    )
    
    test_envs = {'test': test_env}
    env2scans = {'test': ['test']}
    
    return test_envs, env2scans

def main():
    """Main function - real-time path reconstruction system"""
    import argparse
    import sys
    
    print("Starting the real-time path reconstruction system")
    print("Core features:")
    print("  1. Build the topology graph in real time during navigation")
    print("  2. Run a test on n instructions")
    print("  3. Compute navigation performance metrics")
    print("  4. Reconstruct optimal paths with the A* algorithm based on endpoints from user feedback")
    print("  5. Generate two files: user feedback data + A* reconstructed paths")
    
    # Import required modules
    try:
        from r2r.parser import parse_args as parse_agent_args
        from r2r.agent import GMapNavAgent
        from utils.misc import set_random_seed
    except ImportError as e:
        print(f"Failed to import modules: {e}")
        sys.exit(1)
    
    # Parse agent arguments
    args = parse_agent_args()
    
    # Add arguments related to path reconstruction
    reconstruction_parser = argparse.ArgumentParser(add_help=False)
    reconstruction_parser.add_argument('--output_dir', type=str, 
                                     default='path_reconstruction_output',
                                     help='Output directory for path reconstruction results')
    reconstruction_parser.add_argument('--num_episodes', type=int, default=500,
                                     help='Number of test instructions')
    reconstruction_parser.add_argument('--scan', type=str, default=None,
                                     help='Specify a single scan ID')
    reconstruction_parser = argparse.ArgumentParser(add_help=False)
    reconstruction_parser.add_argument('--output_dir', type=str, 
                                     default='path_reconstruction_output')
    reconstruction_parser.add_argument('--num_episodes', type=int, default=500)
    reconstruction_parser.add_argument('--scan', type=str, default=None)
    # New: load-from-cache mode
    reconstruction_parser.add_argument('--use_cache', action='store_true',
                                     help='Use cached topology instead of running navigation')
    reconstruction_parser.add_argument('--cache_file', type=str,
                                     help='Path to cache file for topology')
    reconstruction_parser.add_argument('--feedback_file', type=str,
                                     help='Path to feedback data file (required when using cache)')
    
    recon_args, _ = reconstruction_parser.parse_known_args()
    
    # Merge the arguments into the main args object
    for key, value in vars(recon_args).items():
        if value is not None:
            setattr(args, key, value)
    
    # Make sure we are in test mode
    args.test = True
    
    # Set the random seed
    set_random_seed(args.seed)
    
    # Print key arguments
    print("\n=== System Configuration ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of test instructions: {getattr(args, 'num_episodes', 500)}")
    print(f"Specified scan: {getattr(args, 'scan', 'NOT_SET')}")
    print(f"Connectivity directory: {getattr(args, 'connectivity_dir', 'NOT_SET')}")
    print(f"Feature file: {getattr(args, 'img_ft_file', 'NOT_SET')}")
    print(f"All ScanVP Cands: {getattr(args, 'all_scanvp_cands', 'NOT_SET')}")
    print(f"Pano Inputs Path: {getattr(args, 'pano_inputs_path', 'NOT_SET')}")
    print(f"Anno directory: {getattr(args, 'anno_dir', 'NOT_SET')}")
    print(f"New Anno directory: {getattr(args, 'new_anno_dir', 'NOT_SET')}")
    
    # Build the dataset
    print("\nInitializing environment...")
    test_envs, _ = build_dataset(args, rank=0, is_test=True,
                               specified_scan=getattr(args, 'scan', None))
    
    if not test_envs:
        print("Environment initialization failed")
        sys.exit(1)
    
    # Select environment
    env_name = list(test_envs.keys())[0]
    env = test_envs[env_name]
    
    # Initialize the agent
    print("Initializing navigation agent...")
    agent = GMapNavAgent(args, env, rank=0)
    
    # Load the pretrained model
    if hasattr(args, 'resume_file') and args.resume_file:
        print(f"Loading pretrained model: {args.resume_file}")
        try:
            load_iter = agent.load(args.resume_file)
            print(f"Successfully loaded the model from iteration {load_iter}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Continuing with an untrained model")
    else:
        print("Warning: no pretrained model provided")
    
    print(f"Using environment '{env_name}' for real-time path reconstruction")
    
    # Start real-time path reconstruction
    print("\nStarting real-time path reconstruction task...")
    try:
        feedback_data_path, reconstructed_paths_path = reconstruct_paths_from_realtime_feedback(
            agent, env, args.output_dir
        )
        
        print(f"\nReal-time path reconstruction task complete!")
        print(f"User feedback data: {feedback_data_path}")
        print(f"A* reconstructed paths: {reconstructed_paths_path}")
        # The line below makes the program exit immediately and the batch run ends as well. Why?
        # os._exit(0)
        
    except Exception as e:
        print(f"Path reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()