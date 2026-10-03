from collections import defaultdict
import numpy as np
import torch
import time


MAX_DIST = 30
MAX_STEP = 10

def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return dist

def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    if b[1] < a[1]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz/xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist

def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts


class FloydGraph(object):
    
    def __init__(self):
        # Remove lambda functions entirely; use plain dicts and getter methods
        self._dis = {}
        self._point = {}
        self._visited = set()
    
    def _get_distance(self, x, y):
        """Safely get the distance; return a default value if it does not exist"""
        if x not in self._dis:
            return 95959595
        if y not in self._dis[x]:
            return 95959595
        return self._dis[x][y]
    
    def _get_point(self, x, y):
        """Safely get the path point; return a default value if it does not exist"""
        if x not in self._point:
            return ""
        if y not in self._point[x]:
            return ""
        return self._point[x][y]
    
    def _ensure_node_exists(self, x):
        """Ensure node x exists in the dicts"""
        if x not in self._dis:
            self._dis[x] = {}
        if x not in self._point:
            self._point[x] = {}

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._get_distance(x, y)

    def add_edge(self, x, y, dis):
        self._ensure_node_exists(x)
        self._ensure_node_exists(y)
        
        current_dis = self._get_distance(x, y)
        if dis < current_dis:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        self._ensure_node_exists(k)
        
        # Get all known nodes
        all_nodes = set(self._dis.keys()) | set(self._point.keys())
        
        for x in all_nodes:
            if x == k:
                continue
            self._ensure_node_exists(x)
            
            for y in all_nodes:
                if x != y and y != k:
                    self._ensure_node_exists(y)
                    
                    dist_xk = self._get_distance(x, k)
                    dist_ky = self._get_distance(k, y)
                    current_dist_xy = self._get_distance(x, y)
                    
                    new_dist = dist_xk + dist_ky
                    if new_dist < current_dist_xy:
                        self._dis[x][y] = new_dist
                        self._dis[y][x] = new_dist
                        self._point[x][y] = k
                        self._point[y][x] = k
                        
        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        
        point = self._get_point(x, y)
        if point == "":     # Direct edge
            return [y]
        else:
            k = point
            return self.path(x, k) + self.path(k, y)


class GraphMap(object):
    def __init__(self, start_vp):
        self.start_vp = start_vp    # start viewpoint

        self.node_positions = {}             # viewpoint to position (x, y, z)
        self.graph = FloydGraph()   # shortest path graph
        self.node_embeds = {}       # {viewpoint: feature (sum feature, count)}
        self.node_stop_scores = {}  # {viewpoint: prob}
        self.node_nav_scores = {}   # {viewpoint: {t: prob}}
        self.node_step_ids = {}
        self.rewrite = {}

    def init_episode(self):
        self.node_stop_scores = {}
        self.node_nav_scores = {}
        self.node_step_ids = {}

    def update_graph(self, ob):
        self.node_positions[ob['viewpoint']] = ob['position']
        for cc in ob['candidate']:
            self.node_positions[cc['viewpointId']] = cc['position']
            dist = calc_position_distance(ob['position'], cc['position'])
            self.graph.add_edge(ob['viewpoint'], cc['viewpointId'], dist)
        self.graph.update(ob['viewpoint'])

    def update_node_embed(self, vp, embed, rewrite=False):
        # Make sure embed is detached to avoid gradient computation issues
        if torch.is_tensor(embed):
            embed = embed.detach()
        
        if rewrite:
            self.node_embeds[vp] = [embed, 1]
            self.rewrite[vp] = True
        elif not self.rewrite.get(vp, False):
            if vp in self.node_embeds:
                # Use a non-in-place operation to avoid gradient errors
                self.node_embeds[vp][0] = self.node_embeds[vp][0] + embed
                self.node_embeds[vp][1] += 1
            else:
                self.node_embeds[vp] = [embed, 1]
    
    def get_node_embed(self, vp):
        return self.node_embeds[vp][0] / self.node_embeds[vp][1]

    def get_pos_fts(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, angle_feat_size=4):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.node_positions[cur_vp], self.node_positions[vp],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.graph.distance(cur_vp, vp) / MAX_DIST, \
                    len(self.graph.path(cur_vp, vp)) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def save_to_json(self):
        nodes = {}
        for vp, pos in self.node_positions.items():
            nodes[vp] = {
                'location': pos,    # (x, y, z)
                'visited': self.graph.visited(vp),
            }
            if nodes[vp]['visited']:
                nodes[vp]['stop_prob'] = self.node_stop_scores[vp]['stop']
                nodes[vp]['og_objid'] = self.node_stop_scores[vp]['og']
            else:
                nodes[vp]['nav_prob'] = self.node_nav_scores[vp]

        edges = []
        for k, v in self.graph._dis.items():
            for kk in v.keys():
                edges.append((k, kk))
                
        return {'nodes': nodes, 'edges': edges}
    
class GraphMapWithCache(GraphMap):
    """
    Extends the GraphMap class with caching of raw input data,
    to support feature recomputation during model migration
    """
    def __init__(self, start_vp):
        super().__init__(start_vp)
        self.cached_raw_inputs = {}  # {viewpoint: cached_input_data}
        self.cache_metadata = {
            'model_version': None,
            'feature_dim': None,
            'cached_count': 0
        }
    
    def cache_raw_input(self, viewpoint, pano_input, obs_data):
        """
        Cache a node's raw input data for later feature recomputation
        
        Args:
            viewpoint (str): node ID
            pano_input (dict): input data for VLNBert panorama mode
            obs_data (dict): observation data, including heading, elevation, etc.
        """
        # Validate input data integrity
        required_pano_keys = ['view_img_fts', 'loc_fts', 'nav_types', 'view_lens', 'cand_vpids']
        for key in required_pano_keys:
            if key not in pano_input:
                raise ValueError(f"Missing required pano_input key: {key}")
        
        required_obs_keys = ['viewpoint', 'heading', 'elevation', 'position', 'scan']
        for key in required_obs_keys:
            if key not in obs_data:
                raise ValueError(f"Missing required obs_data key: {key}")
        
        # Make sure the data is on the CPU and detached
        cached_data = {
            'pano_input': {
                'view_img_fts': pano_input['view_img_fts'].detach().cpu() if torch.is_tensor(pano_input['view_img_fts']) else pano_input['view_img_fts'],
                'loc_fts': pano_input['loc_fts'].detach().cpu() if torch.is_tensor(pano_input['loc_fts']) else pano_input['loc_fts'],
                'nav_types': pano_input['nav_types'].detach().cpu() if torch.is_tensor(pano_input['nav_types']) else pano_input['nav_types'],
                'view_lens': pano_input['view_lens'].detach().cpu() if torch.is_tensor(pano_input['view_lens']) else pano_input['view_lens'],
                'cand_vpids': [cand_list.copy() if isinstance(cand_list, list) else cand_list for cand_list in pano_input['cand_vpids']]
            },
            'obs_data': {
                'viewpoint': obs_data['viewpoint'],
                'heading': float(obs_data['heading']),
                'elevation': float(obs_data['elevation']), 
                'position': tuple(obs_data['position']) if isinstance(obs_data['position'], (list, tuple)) else obs_data['position'],
                'scan': obs_data['scan'],
                'instr_id': obs_data.get('instr_id', 'unknown')
            },
            'timestamp': time.time(),  # Record the cache time
            'node_type': 'visited' if viewpoint in self.node_embeds else 'candidate'
        }
        
        # Store the cached data
        self.cached_raw_inputs[viewpoint] = cached_data
        self.cache_metadata['cached_count'] += 1
        
        # Optional: record feature dimension info
        if self.cache_metadata['feature_dim'] is None and torch.is_tensor(pano_input['view_img_fts']):
            self.cache_metadata['feature_dim'] = pano_input['view_img_fts'].shape[-1]
    
    def get_cached_input(self, viewpoint):
        """Get the cached input data for the given node"""
        if viewpoint not in self.cached_raw_inputs:
            return None
        return self.cached_raw_inputs[viewpoint]
    
    def has_cached_input(self, viewpoint):
        """Check whether the given node has cached data"""
        return viewpoint in self.cached_raw_inputs
    
    def get_cache_stats(self):
        """Get cache statistics"""
        visited_count = sum(1 for data in self.cached_raw_inputs.values() 
                          if data['node_type'] == 'visited')
        candidate_count = sum(1 for data in self.cached_raw_inputs.values() 
                            if data['node_type'] == 'candidate')
        
        return {
            'total_cached': len(self.cached_raw_inputs),
            'visited_nodes': visited_count,
            'candidate_nodes': candidate_count,
            'feature_dim': self.cache_metadata['feature_dim'],
            'model_version': self.cache_metadata['model_version']
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cached_raw_inputs.clear()
        self.cache_metadata['cached_count'] = 0
    
    def save_with_cache(self, filepath):
        """Save the GraphMap and cached data to a file"""
        save_data = {
            'graph_map': {
                'start_vp': self.start_vp,
                'node_positions': self.node_positions,
                'node_embeds': self.node_embeds,
                'node_stop_scores': self.node_stop_scores,
                'node_nav_scores': self.node_nav_scores,
                'node_step_ids': self.node_step_ids,
                'rewrite': dict(self.rewrite),
                'graph_distances': self.graph._dis,
                'graph_points': self.graph._point,
                'graph_visited': self.graph._visited
            },
            'cached_inputs': self.cached_raw_inputs,
            'cache_metadata': self.cache_metadata,
            'save_timestamp': time.time()
        }
        
        # Use compression when saving to reduce space
        import pickle
        import gzip
        
        with gzip.open(filepath + '.gz', 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved GraphMap with {len(self.cached_raw_inputs)} cached inputs to {filepath}.gz")
    
    @classmethod
    def load_with_cache(cls, filepath):
        """Load the GraphMap and cached data from a file"""
        import pickle
        import gzip
        
        # Try to load a compressed file
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                save_data = pickle.load(f)
        else:
            # Try to load the compressed version
            try:
                with gzip.open(filepath + '.gz', 'rb') as f:
                    save_data = pickle.load(f)
            except:
                # Fall back to a plain pickle file
                with open(filepath, 'rb') as f:
                    save_data = pickle.load(f)
        
        # Rebuild the GraphMap object
        graph_data = save_data['graph_map']
        gmap = cls(graph_data['start_vp'])
        
        # Restore all data
        gmap.node_positions = graph_data['node_positions']
        gmap.node_embeds = graph_data['node_embeds']
        gmap.node_stop_scores = graph_data['node_stop_scores']
        gmap.node_nav_scores = graph_data['node_nav_scores']
        gmap.node_step_ids = graph_data['node_step_ids']
        gmap.rewrite = defaultdict(lambda: False, graph_data['rewrite'])
        
        # Restore the graph structure
        gmap.graph._dis = graph_data['graph_distances']
        gmap.graph._point = graph_data['graph_points']
        gmap.graph._visited = graph_data['graph_visited']
        
        # Restore the cached data
        gmap.cached_raw_inputs = save_data['cached_inputs']
        gmap.cache_metadata = save_data['cache_metadata']
        
        print(f"Loaded GraphMap with {len(gmap.cached_raw_inputs)} cached inputs")
        return gmap
    
    def validate_cache_integrity(self):
        """Validate the integrity of the cached data"""
        errors = []
        
        for viewpoint, cached_data in self.cached_raw_inputs.items():
            # Check the data structure
            if 'pano_input' not in cached_data or 'obs_data' not in cached_data:
                errors.append(f"Missing data structure for viewpoint {viewpoint}")
                continue
            
            pano_input = cached_data['pano_input']
            obs_data = cached_data['obs_data']
            
            # Check required keys
            required_pano_keys = ['view_img_fts', 'loc_fts', 'nav_types', 'view_lens', 'cand_vpids']
            for key in required_pano_keys:
                if key not in pano_input:
                    errors.append(f"Missing pano_input key '{key}' for viewpoint {viewpoint}")
            
            required_obs_keys = ['viewpoint', 'heading', 'elevation', 'position', 'scan']
            for key in required_obs_keys:
                if key not in obs_data:
                    errors.append(f"Missing obs_data key '{key}' for viewpoint {viewpoint}")
            
            # Check tensor data types
            for key in ['view_img_fts', 'loc_fts', 'nav_types', 'view_lens']:
                if key in pano_input and torch.is_tensor(pano_input[key]):
                    if pano_input[key].device.type != 'cpu':
                        errors.append(f"Tensor {key} for viewpoint {viewpoint} not on CPU")
        
        if errors:
            print(f"Cache validation found {len(errors)} errors:")
            for error in errors[:5]:  # Only print the first 5 errors
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        else:
            print(f"Cache validation passed for {len(self.cached_raw_inputs)} cached items")
        
        return len(errors) == 0