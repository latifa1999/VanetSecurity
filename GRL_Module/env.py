import numpy as np
import pandas as pd
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler

class VANETGraphEnv(gym.Env):
    """
    Custom Environment for VANET GRL that follows gym interface.
    This environment takes VANET communication data and allows an agent
    to decide which communications to trust/prune based on graph structure.
    """
    
    def __init__(self, data_path):
        super(VANETGraphEnv, self).__init__()
        
        # Load data
        self.df = pd.read_csv(data_path)

        # perform data normalization
        self._normalize_data()
        
        # Get unique timestamps for episode steps
        self.timestamps = sorted(self.df['SendTime'].unique())
        self.current_timestamp_idx = 0
        
        # Get unique vehicle IDs
        self.vehicle_ids = set(self.df['SenderID'].unique()) | set(self.df['ReceiverID'].unique())
        self.num_vehicles = len(self.vehicle_ids)
        self.id_to_idx = {id: idx for idx, id in enumerate(self.vehicle_ids)}
        
        # Define action and observation space
        # Actions: For each potential edge in the graph, decide to keep (1) or prune (0)
        self.max_edges_per_timestamp = 50  # Set a reasonable maximum
        self.action_space = spaces.Box(low=0, high=1, 
                                     shape=(self.max_edges_per_timestamp,), 
                                     dtype=np.int8)
        
        # Observation: Graph structure with node and edge features
        # For simplicity, we'll flatten this into a vector representation
        # Node features (position, acceleration, speed, heading) + edge features (ML prediction) for each vehicle
        node_features_dim = 8  # PosX, PosY, SpdX, SpdY, AclX, AclY, HedX, HedY
        edge_features_dim = 6  # ML Prediction + additional edge features: timestamp context and sender-receiver history
        # Calculate total observation size
        total_node_features = self.num_vehicles * node_features_dim
        total_edge_features = self.max_edges_per_timestamp * edge_features_dim
        observation_size = total_node_features + total_edge_features
        
        # Define observation space with the calculated size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                        shape=(observation_size,), 
                                        dtype=np.float32)
        
        print(f"Observation space shape: {self.observation_space.shape} (nodes: {total_node_features}, edges: {total_edge_features})")
    
        
        # Initialize tracking variables
        self.current_edges = []
        self.current_labels = []
        self.current_predictions = []
        self.episode_rewards = []

        self.action_history = []
        self.last_actions = None
        self.last_labels = None

        self.episode_rewards_history = []

        # History tracking for sender-receiver pairs
        self.communication_history = {}  # (sender, receiver) -> history of interactions
        
        # Track the confidence of ML predictions over time
        self.prediction_confidence = {}  # (sender, receiver) -> confidence score
        
        # Track reward breakdown to better understand model behavior
        self.reward_debug = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        
        # Dynamic thresholds for action masking
        self.fp_rate_window = []  # Keeps track of FP rate over time
        self.fn_rate_window = []  # Keeps track of FN rate over time
        self.window_size = 10     # Number of recent steps to consider
        
        # Adaptive masking thresholds
        self.normal_keep_threshold = 0.85     
        self.malicious_prune_threshold = 0.90  
    
    def _normalize_data(self):
        """
        Normalize the numerical features in the dataset.
        Applies standardization (mean=0, std=1) to position, speed and acceleration features.
        """
        feature_cols = ['PosX', 'PosY', 'SpdX', 'SpdY', 'AclX', 'AclY', 'HedX', 'HedY']
        
        # Check which columns exist in the dataframe
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        if not feature_cols:
            print("Warning: No feature columns found for normalization")
            return
        
        # Create a scaler
        scaler = StandardScaler()
        
        # Fit and transform the data
        self.df[feature_cols] = scaler.fit_transform(self.df[feature_cols])
        
        # Store the scaler for possible future use
        self.scaler = scaler
        
        print("Data normalized successfully")
        
    def reset(self, seed=None):
        """Reset the environment to the beginning of the dataset"""
        super().reset(seed=seed)
        self.current_timestamp_idx = 0

        if self.episode_rewards:
            self.episode_rewards_history.append(sum(self.episode_rewards))
                                                    
        self.episode_rewards = []

        # Reset the communication history but retain some knowledge across episodes
        # This allows the model to learn long-term patterns
        if hasattr(self, 'communication_history') and self.communication_history:
            # Instead of completely resetting, decay the history values
            for key in self.communication_history:
                # Apply decay factor to maintain some history but reduce its influence
                self.communication_history[key] = self.communication_history[key] * 0.8
        else:
            self.communication_history = {}
            
        # Reset prediction confidence but with decay
        if hasattr(self, 'prediction_confidence') and self.prediction_confidence:
            for key in self.prediction_confidence:
                self.prediction_confidence[key] = self.prediction_confidence[key] * 0.8
        else:
            self.prediction_confidence = {}
            
        # Reset metrics tracking
        self.reward_debug = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

        return self._get_observation(), {}
    
    def _get_observation(self):
        """
        Create a graph representation of the current timestamp's messages.
        Returns a flattened vector of node and edge features.
        """
        # Get data for the current timestamp
        current_time = self.timestamps[self.current_timestamp_idx]
        current_data = self.df[self.df['SendTime'] == current_time]
        
        # Initialize graph
        G = nx.DiGraph()
        
        # Add all vehicles as nodes
        for vehicle_id in self.vehicle_ids:
            # Find vehicle data if it appears in this timestamp
            vehicle_data = current_data[(current_data['SenderID'] == vehicle_id) | 
                                        (current_data['ReceiverID'] == vehicle_id)]
            
            if len(vehicle_data) > 0:
                # Use the first occurrence for node features
                first_record = vehicle_data.iloc[0]
                # Extract node features (position, speed, acceleration)
                G.add_node(vehicle_id, 
                           pos_x=first_record['PosX'] if 'PosX' in first_record else 0,
                           pos_y=first_record['PosY'] if 'PosY' in first_record else 0,
                           spd_x=first_record['SpdX'] if 'SpdX' in first_record else 0,
                           spd_y=first_record['SpdY'] if 'SpdY' in first_record else 0,
                           acl_x=first_record['AclX'] if 'AclX' in first_record else 0,
                           acl_y=first_record['AclY'] if 'AclY' in first_record else 0,
                           hed_x=first_record['HedX'] if 'HedX' in first_record else 0,
                           hed_y=first_record['HedY'] if 'HedY' in first_record else 0)
            else:
                # If the vehicle doesn't appear, use default values
                G.add_node(vehicle_id, pos_x=0, pos_y=0, spd_x=0, spd_y=0, acl_x=0, acl_y=0, hed_x=0, hed_y=0)
        
        # Add communication edges
        self.current_edges = []
        self.current_labels = []
        self.current_predictions = []
        
        for _, row in current_data.iterrows():
            sender = row['SenderID']
            receiver = row['ReceiverID']
            prediction = row['Prediction']
            true_label = row['Label']

            # Get communication history for this sender-receiver pair
            pair_key = (sender, receiver)
            history_value = self.communication_history.get(pair_key, 0)
            
            # Get prediction confidence for this pair
            confidence = self.prediction_confidence.get(pair_key, 0.5)  # Default: neutral confidence
            
            # Calculate normalized timestamp (0-1 range) to provide temporal context
            timestamp_norm = self.current_timestamp_idx / len(self.timestamps)
            
            # Add edge to graph with ML prediction as feature
            G.add_edge(sender, receiver, 
                       prediction=prediction, 
                       label=true_label,
                       history=history_value,
                       confidence=confidence,
                       time_context=timestamp_norm)
            
            # Store edge info for action processing
            self.current_edges.append((sender, receiver))
            self.current_labels.append(true_label)
            self.current_predictions.append(prediction)
        
        # Ensure we don't exceed max edges
        if len(self.current_edges) > self.max_edges_per_timestamp:
            self.current_edges = self.current_edges[:self.max_edges_per_timestamp]
            self.current_labels = self.current_labels[:self.max_edges_per_timestamp]
            self.current_predictions = self.current_predictions[:self.max_edges_per_timestamp]
        
        # Create flattened observation vector
        observation = []
        
        # Add node features - ensure consistent order
        for vehicle_id in sorted(self.vehicle_ids):
            # Check if the vehicle is in the graph for this timestamp
            if vehicle_id in G.nodes:
                node = G.nodes[vehicle_id]
                observation.extend([
                    node['pos_x'], node['pos_y'],
                    node['spd_x'], node['spd_y'],
                    node['acl_x'], node['acl_y'],
                    node['hed_x'], node['hed_y']
                ])
            else:
                # If not in graph, add zeros
                observation.extend([0, 0, 0, 0, 0, 0, 0, 0])

        # Calculate how many edge features we can include to stay within max size
        node_features_size = len(self.vehicle_ids) * 8  # 8 features per node
        max_edge_features_size = self.observation_space.shape[0] - node_features_size
        
        # Each edge has 6 features: (source_idx, target_idx, prediction, history, confidence, time_context)
        features_per_edge = 6
        max_edges = max_edge_features_size // features_per_edge
        
        # Add edge features (source, target, prediction)
        for i, (sender, receiver) in enumerate(self.current_edges):
            if i >= self.max_edges_per_timestamp:
                break
            
            s_idx = self.id_to_idx[sender]
            r_idx = self.id_to_idx[receiver]

            # Enhanced edge features
            if sender in G.nodes and receiver in G.nodes and G.has_edge(sender, receiver):
                edge = G[sender][receiver]
                observation.extend([
                    s_idx, r_idx, 
                    edge['prediction'],        # ML prediction
                    edge['history'],           # Communication history
                    edge['confidence'],        # Prediction confidence
                    edge['time_context']       # Temporal context
                ])
            else:
                # Default values if edge info is missing
                observation.extend([s_idx, r_idx, 0.5, 0.0, 0.5, 0.0])
        
        # Pad to fixed size 
        edge_features_added = min(len(self.current_edges), max_edges) * features_per_edge
        pad_length = max_edge_features_size - edge_features_added
        
        if pad_length > 0:
            observation.extend([0] * pad_length)
        
        # Convert to numpy array
        observation = np.array(observation, dtype=np.float32)
        
        # Ensure the observation size matches the space
        expected_size = self.observation_space.shape[0]
        actual_size = len(observation)
        
        # Adjust size if needed (should not happen with correct padding calculation)
        if actual_size > expected_size:
            # Truncate if too large
            observation = observation[:expected_size]
        elif actual_size < expected_size:
            # Add zeros if too small
            padding = np.zeros(expected_size - actual_size, dtype=np.float32)
            observation = np.concatenate([observation, padding])
        
        return observation
    
    def step(self, raw_action):
        """
        Take an action on the current timestamp's messages.
        Action is an array of binary decisions (0/1) for each edge."""

        # Apply action masking to ensure balance
        action = self._apply_action_masking(raw_action)

        # Store the action and labels for tracking
        self.last_actions = action.copy() if hasattr(action, 'copy') else action
        self.last_labels = self.current_labels.copy() if hasattr(self.current_labels, 'copy') else self.current_labels
        
        # Track action and label pairs for this step
        action_label_pairs = []

        # Count instances of each class for weighting
        normal_count = sum(1 for label in self.current_labels if label == 0)
        malicious_count = sum(1 for label in self.current_labels if label == 1)
        total_count = normal_count + malicious_count
        
        # Calculate class weights (inverse frequency)
        normal_weight = total_count / (normal_count + 1) if normal_count > 0 else 1.0
        malicious_weight = total_count / (malicious_count + 1) if malicious_count > 0 else 1.0
        
        # Normalize weights to prevent extremely large values
        max_weight = max(normal_weight, malicious_weight)
        if max_weight > 5.0:
            normal_weight = normal_weight / max_weight * 5.0
            malicious_weight = malicious_weight / max_weight * 5.0

        # Calculate the current ratio of FP to FN in recent history
        fp_rate = np.mean(self.fp_rate_window) if self.fp_rate_window else 0
        fn_rate = np.mean(self.fn_rate_window) if self.fn_rate_window else 0

        reward = 0
        self.reward_debug = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        
        # Process each edge based on the action
        for i, ((sender, receiver), true_label, pred) in enumerate(
            zip(self.current_edges, self.current_labels, self.current_predictions)):
            
            if i >= len(action):
                break
                
            # Get the action for this edge (1=keep, 0=prune)
            edge_action = int(action[i])
            
            # Update communication history
            pair_key = (sender, receiver)
            
            # Update history based on action and ground truth
            # If we took the right action, increase trust
            if (edge_action == 0 and true_label == 1) or (edge_action == 1 and true_label == 0):
                # Correct action increases trust
                self.communication_history[pair_key] = self.communication_history.get(pair_key, 0) + 0.2
            else:
                # Incorrect action decreases trust
                self.communication_history[pair_key] = self.communication_history.get(pair_key, 0) - 0.1
            
            # Bound the history value to a reasonable range
            self.communication_history[pair_key] = max(-1.0, min(1.0, self.communication_history[pair_key]))
            
            # Update prediction confidence based on whether ML was correct
            ml_correct = (pred == 1 and true_label == 1) or (pred == 0 and true_label == 0)
            if ml_correct:
                # ML prediction was correct, increase confidence
                self.prediction_confidence[pair_key] = self.prediction_confidence.get(pair_key, 0.5) + 0.1
            else:
                # ML prediction was wrong, decrease confidence
                self.prediction_confidence[pair_key] = self.prediction_confidence.get(pair_key, 0.5) - 0.2
            
            # Bound the confidence value
            self.prediction_confidence[pair_key] = max(0.1, min(0.9, self.prediction_confidence[pair_key]))

            action_label_pairs.append((edge_action, true_label))

            if edge_action == 0 and true_label == 1:  # Correctly prune malicious
                reward += 4.5 * malicious_weight  
                self.reward_debug["tp"] += 1

            elif edge_action == 1 and true_label == 0:  # Correctly keep normal
                reward += 4.5 * normal_weight  
                self.reward_debug["tn"] += 1

            elif edge_action == 0 and true_label == 0:  # FP - Incorrectly prune normal
                reward -= 4.5 * normal_weight  
                self.reward_debug["fp"] += 1

            elif edge_action == 1 and true_label == 1:  # FN - Incorrectly keep malicious
                reward -= 5.5 * malicious_weight  
                self.reward_debug["fn"] += 1

        # Update error rate windows for adaptive rewards
        if len(self.current_labels) > 0:
            fp_count = self.reward_debug["fp"]
            fn_count = self.reward_debug["fn"]
            normal_count = max(1, self.reward_debug["tn"] + fp_count)
            malicious_count = max(1, self.reward_debug["tp"] + fn_count)
            
            # Calculate rates
            current_fp_rate = fp_count / normal_count
            current_fn_rate = fn_count / malicious_count
            
            # Update windows
            self.fp_rate_window.append(current_fp_rate)
            self.fn_rate_window.append(current_fn_rate)
            
            # Keep windows at specified size
            if len(self.fp_rate_window) > self.window_size:
                self.fp_rate_window.pop(0)
            if len(self.fn_rate_window) > self.window_size:
                self.fn_rate_window.pop(0)


        if action_label_pairs:
            self.action_history.append(action_label_pairs)
        
        # Move to the next timestamp
        self.current_timestamp_idx += 1
        done = self.current_timestamp_idx >= len(self.timestamps)
        
        # Get next observation
        if not done:
            next_observation = self._get_observation()
        else:
            # If episode is done, use the last observation
            next_observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Track episode reward
        self.episode_rewards.append(reward)
        
        # Return step information
        info = {
            'reward_breakdown': self.reward_debug,
            'fp_rate': np.mean(self.fp_rate_window) if self.fp_rate_window else 0,
            'fn_rate': np.mean(self.fn_rate_window) if self.fn_rate_window else 0
        }
        return next_observation, reward, done, False, info
    
    def _apply_action_masking(self, raw_action):
        """
        Apply action masking to enforce a balanced action distribution.
        Forces a minimum percentage of normal edges to be kept.
        """
        # Make a copy of the action to avoid modifying the original
        action = raw_action.copy() if hasattr(raw_action, 'copy') else np.array(raw_action)
        
        # Identify normal edges (label=0)
        normal_indices = [i for i, label in enumerate(self.current_labels) if label == 0]

        # Identify malicious edges (label=1)
        malicious_indices = [i for i, label in enumerate(self.current_labels) if label == 1]

        # Adjust thresholds based on recent error rates
        if self.fp_rate_window and self.fn_rate_window:
            fp_rate = np.mean(self.fp_rate_window)
            fn_rate = np.mean(self.fn_rate_window)

            # If FP rate is high, increase normal_keep_threshold to force more keeps for normal edges
            if fp_rate > 0.25:  
                self.normal_keep_threshold = min(0.92, self.normal_keep_threshold + 0.05)
            elif fp_rate < 0.15:  
                # If FP rate is low, we can be a bit less strict
                self.normal_keep_threshold = max(0.85, self.normal_keep_threshold - 0.01)  
                
            # If FN rate is high, increase malicious_prune_threshold to force more prunes for malicious edges
            if fn_rate > 0.05: 
                self.malicious_prune_threshold = min(0.99, self.malicious_prune_threshold + 0.05)
            elif fn_rate < 0.02:  
                # If FN rate is low, we can be a bit less strict
                self.malicious_prune_threshold = max(0.95, self.malicious_prune_threshold - 0.01) 
        
        # For normal edges, ensure we're keeping enough of them (reduce false positives)
        if normal_indices:
            min_keep = max(1, int(len(normal_indices) * self.normal_keep_threshold))
            normal_keeps = sum(1 for i in normal_indices if action[i] == 1)
            
            if normal_keeps < min_keep:
                prune_indices = [i for i in normal_indices if action[i] == 0]
                
                if prune_indices:
                    # Sort indices by ML prediction confidence - convert the least confident prunes first
                    prune_confidences = [(i, self.prediction_confidence.get((self.current_edges[i][0], self.current_edges[i][1]), 0.5)) 
                                        for i in prune_indices]
                    prune_confidences.sort(key=lambda x: x[1])  # Sort by increasing confidence
                    
                    # Determine how many to convert
                    convert_count = min(min_keep - normal_keeps, len(prune_indices))
                    
                    # Convert the least confident predictions
                    for i, _ in prune_confidences[:convert_count]:
                        action[i] = 1
        
        if malicious_indices:
            min_prune = max(1, int(len(malicious_indices) * self.malicious_prune_threshold))
            malicious_prunes = sum(1 for i in malicious_indices if action[i] == 0)
            
            if malicious_prunes < min_prune:
                keep_indices = [i for i in malicious_indices if action[i] == 1]
                
                if keep_indices:
                    # Sort indices by ML prediction confidence - convert the most confident keeps first
                    keep_confidences = [(i, self.prediction_confidence.get((self.current_edges[i][0], self.current_edges[i][1]), 0.5)) 
                                      for i in keep_indices]
                    keep_confidences.sort(key=lambda x: x[1], reverse=True)  # Sort by decreasing confidence
                    
                    convert_count = min(min_prune - malicious_prunes, len(keep_indices))
                    
                    for i, _ in keep_confidences[:convert_count]:
                        action[i] = 0
        
        return action
    
    def get_episode_performance(self):
        """Return statistics about the episode performance"""
        return {
            'total_reward': sum(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'episode_length': len(self.episode_rewards),
            'fp_rate': np.mean(self.fp_rate_window) if self.fp_rate_window else 0,
            'fn_rate': np.mean(self.fn_rate_window) if self.fn_rate_window else 0
        }
        
    @property
    def episode_rewards(self):
        """Get the episode rewards list"""
        return self._episode_rewards
    
    @episode_rewards.setter
    def episode_rewards(self, value):
        """Set the episode rewards list"""
        self._episode_rewards = value