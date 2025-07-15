#!/usr/bin/env python3

import os
import socket
import struct
import torch
import numpy as np
import pandas as pd
import time
from argparse import Namespace
from exp.exp_classification import Exp_Classification
import traceback
import signal
import sys

# Import stable-baselines3 for GRL model loading
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class MessageData:
    def __init__(self, data):
        try:
            current_pos = 0
            
            # 1) sender_id (100 bytes)
            self.sender_id = data[current_pos:current_pos+100].split(b'\0')[0].decode('utf-8', errors='ignore')
            current_pos += 100
            
            # 2) receiver_id (100 bytes)
            self.receiver_id = data[current_pos:current_pos+100].split(b'\0')[0].decode('utf-8', errors='ignore')
            current_pos += 100
            
            # 3) doubles (80 bytes, 10 doubles)
            doubles_data = data[current_pos:current_pos+80]
            nums = struct.unpack('10d', doubles_data)
            current_pos += 80
            
            # Store only the necessary features in order
            self.pos_x, self.pos_y = nums[0], nums[1]
            self.spd_x, self.spd_y = nums[2], nums[3]
            self.acl_x, self.acl_y = nums[4], nums[5]
            self.hed_x, self.hed_y = nums[6], nums[7]
            self.simulation_time = nums[8]
            
            # 4) attack_type (42 bytes)
            attack_type_size = 42
            self.attack_type = data[current_pos:current_pos+attack_type_size].split(b'\0')[0].decode('utf-8', errors='ignore')
            current_pos += attack_type_size
            
            # 5) is_malicious (4 bytes)
            # Check if we have enough data for is_malicious
            self.is_malicious = 0  # Default value
            remaining_bytes = len(data) - current_pos
            
            if remaining_bytes >= 4:
                malicious_data = data[current_pos:current_pos+4]
                if len(malicious_data) == 4:
                    self.is_malicious = struct.unpack('i', malicious_data)[0]
                    print(f"Successfully unpacked is_malicious: {self.is_malicious}")
                else:
                    print(f"Malicious data wrong length: {len(malicious_data)}")
            else:
                # Try looking for is_malicious at the end of the data
                if remaining_bytes > 0:
                    print(f"Trying to extract is_malicious from the last {remaining_bytes} bytes")
                    try:
                        # If we have at least 1 byte, interpret as boolean (0 or 1)
                        last_byte = data[-1]
                        self.is_malicious = 1 if last_byte > 0 else 0
                        print(f"Extracted is_malicious from last byte: {self.is_malicious}")
                    except:
                        print("Failed to extract from last byte")
                else:
                    print(f"Not enough data for is_malicious field")

            # Map sender_id and receiver_id to ints for GRL
            try:
                self.sender_int = int(''.join(filter(str.isdigit, self.sender_id)))
                self.receiver_int = int(''.join(filter(str.isdigit, self.receiver_id)))
            except:
                # Fallback if ID extraction fails
                self.sender_int = hash(self.sender_id) % 1000
                self.receiver_int = hash(self.receiver_id) % 1000
                
            # Print detailed info for debugging
            print(f"Message data:")
            print(f"Sender: {self.sender_id} -> Receiver: {self.receiver_id}")
            print(f"Position: ({self.pos_x}, {self.pos_y})")
            print(f"Speed: ({self.spd_x}, {self.spd_y})")
            print(f"Accel: ({self.acl_x}, {self.acl_y})")
            print(f"Heading: ({self.hed_x}, {self.hed_y})")
            print(f"Attack type: {self.attack_type}")
            print(f"Is malicious: {self.is_malicious}")
                
        except Exception as e:
            print(f"Error unpacking data: {e}")
            traceback.print_exc()
            raise

class DataProcessor:
    def __init__(self, initial_dataset_path, switch_threshold=100):
        self.switch_threshold = switch_threshold
        self.collected_data = []
        self.feature_columns = ['posx', 'posy', 'spdx', 'spdy', 'aclx', 'acly', 'hedx', 'hedy']
        
        # Load and store finetuning data statistics
        self.initialize_from_dataset(initial_dataset_path)
        
    def initialize_from_dataset(self, dataset_path):
        """Initialize statistics from finetuning dataset"""
        df = pd.read_csv(dataset_path)
        features = df[self.feature_columns]
        
        # Store finetuning statistics
        self.finetune_means = features.mean()
        self.finetune_stds = features.std()
        self.ranges = {
            col: (features[col].min(), features[col].max())
            for col in self.feature_columns
        }
        
        print("Finetuning data statistics loaded")
        print(f"Feature means: {self.finetune_means}")
        print(f"Feature stds: {self.finetune_stds}")
        
    def get_simulation_statistics(self):
        """Calculate statistics from collected simulation data"""
        if len(self.collected_data) < self.switch_threshold:
            return None, None
            
        sim_df = pd.DataFrame(self.collected_data, columns=self.feature_columns)
        means = sim_df.mean()
        stds = sim_df.std()
        
        # Handle zero standard deviations
        zero_stds = stds == 0
        if zero_stds.any():
            print(f"Zero standard deviation in simulation data for features: {stds[zero_stds].index.tolist()}")
            stds[zero_stds] = self.finetune_stds[zero_stds]
        
        return means, stds
    
    def process_row(self, msg_data):
        """Process a single row of data"""
        # Create features dictionary
        features_dict = {
            'posx': msg_data.pos_x, 'posy': msg_data.pos_y,
            'spdx': msg_data.spd_x, 'spdy': msg_data.spd_y,
            'aclx': msg_data.acl_x, 'acly': msg_data.acl_y,
            'hedx': msg_data.hed_x, 'hedy': msg_data.hed_y
        }
        
        # Add to collected data
        self.collected_data.append(features_dict)
        
        features = pd.DataFrame([features_dict])
        label = np.array([msg_data.is_malicious])
        
        # Print raw features for debugging
        print("Raw features before clipping:")
        for col in self.feature_columns:
            print(f"  {col}: {features[col].iloc[0]:.6f}")
        
        # Clip values to ranges from finetuning data
        for col in self.feature_columns:
            min_val, max_val = self.ranges[col]
            if features[col].iloc[0] < min_val or features[col].iloc[0] > max_val:
                print(f"Clipping {col} from {features[col].iloc[0]:.4f} to [{min_val:.4f}, {max_val:.4f}]")
            features[col] = features[col].clip(min_val, max_val)
        
        # Choose normalization statistics
        sim_means, sim_stds = self.get_simulation_statistics()
        if sim_means is not None:
            means, stds = sim_means, sim_stds
        else:
            means, stds = self.finetune_means, self.finetune_stds
        
        # Normalize features
        normalized = (features - means) / stds
        
        # Print normalized features for debugging
        print("Normalized features:")
        for i, col in enumerate(self.feature_columns):
            norm_val = normalized.values[0, i]
            print(f"  {col}: {norm_val:.6f}")
        
        return normalized.values, label

class GRLEvaluator:
    """
    Handles GRL model loading and evaluation with action masking.
    Loads the model once and keeps it in memory for fast evaluation.
    """
    
    def __init__(self, grl_model_path, log_dir="grl_logs"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "online_messages.csv")
        self.action_cache = {}  # Cache for storing previous decisions
        
        # Initialize tracking for FP/FN rates for adaptive masking
        self.fp_rate_window = []
        self.fn_rate_window = []
        self.window_size = 10
        
        # Action masking thresholds
        self.normal_keep_threshold = 0.85
        self.malicious_prune_threshold = 0.98
        
        # Track recent decisions for calculating error rates
        self.recent_decisions = {
            "tp": 0, "fp": 0, "tn": 0, "fn": 0
        }
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                header = "timestamp,sender_id,receiver_id,pos_x,pos_y,spd_x,spd_y,acl_x,acl_y,hed_x,hed_y,ml_pred,ml_prob,raw_grl_action,masked_grl_action,is_malicious\n"
                f.write(header)
        
        # Load the GRL model
        print(f"Loading GRL model from {grl_model_path}")
        try:
            # Load the model using stable-baselines3
            self.model = PPO.load(grl_model_path)
            print("GRL model loaded successfully")
            
            # Prepare a dummy environment to format observations
            self.dummy_env = None
            self.vehicle_ids_map = {}
            self.max_vehicle_id = 0
            
            print("GRL evaluator initialized successfully")
        except Exception as e:
            print(f"Error loading GRL model: {e}")
            traceback.print_exc()
            raise
    
    def prepare_observation(self, msg_data, ml_pred, ml_prob):
        """
        Format the message data into an observation that the GRL model can use.
        Uses a simplified format to avoid having to create a full VANETGraphEnv instance.
        """
        # Create a simplified observation focused on the current message
        features = []
        
        # Map vehicle IDs to indices if not seen before
        sender_id = msg_data.sender_id
        receiver_id = msg_data.receiver_id
        
        if sender_id not in self.vehicle_ids_map:
            self.vehicle_ids_map[sender_id] = self.max_vehicle_id
            self.max_vehicle_id += 1
        
        if receiver_id not in self.vehicle_ids_map:
            self.vehicle_ids_map[receiver_id] = self.max_vehicle_id
            self.max_vehicle_id += 1
        
        sender_idx = self.vehicle_ids_map[sender_id]
        receiver_idx = self.vehicle_ids_map[receiver_id]
        
        # Add all features to the observation
        features = [
            # Node features (position, speed, acceleration)
            msg_data.pos_x, msg_data.pos_y,
            msg_data.spd_x, msg_data.spd_y,
            msg_data.acl_x, msg_data.acl_y,
            msg_data.hed_x, msg_data.hed_y,
            
            # Edge features
            sender_idx, receiver_idx,
            ml_pred,  # ML prediction
            ml_prob,  # ML confidence
            0.5,      # Default value for history
            0.0       # Default value for time context
        ]
        
        # Pad to expected size (460 is the size used in the training environment)
        # This matches the expected input size for the GRL model
        expected_size = 460  # Adjust this to match your model's input size
        
        # Pad with zeros if needed
        if len(features) < expected_size:
            features.extend([0.0] * (expected_size - len(features)))
        # Truncate if too large (shouldn't happen with our format)
        elif len(features) > expected_size:
            features = features[:expected_size]
        
        # Convert to numpy array with the correct shape and dtype
        observation = np.array(features, dtype=np.float32)
        
        return observation
    
    def apply_action_masking(self, raw_action, is_malicious, ml_pred):
        """
        Apply action masking to ensure balanced decisions.
        Forces a minimum percentage of normal messages to be kept and
        malicious messages to be pruned.
        """
        # For ML predictions with high probability, trust ML
        # This ensures we don't end up pruning everything
        
        # Adjust thresholds based on recent error rates
        if self.fp_rate_window and self.fn_rate_window:
            fp_rate = np.mean(self.fp_rate_window)
            fn_rate = np.mean(self.fn_rate_window)
            
            # If FP rate is high, increase normal_keep_threshold to keep more normal messages
            if fp_rate > 0.25:
                self.normal_keep_threshold = min(0.92, self.normal_keep_threshold + 0.05)
            elif fp_rate < 0.15:
                self.normal_keep_threshold = max(0.85, self.normal_keep_threshold - 0.01)
                
            # If FN rate is high, increase malicious_prune_threshold to prune more malicious
            if fn_rate > 0.05:
                self.malicious_prune_threshold = min(0.99, self.malicious_prune_threshold + 0.05)
            elif fn_rate < 0.02:
                self.malicious_prune_threshold = max(0.95, self.malicious_prune_threshold - 0.01)
        
        # Apply masking based on message type
        if ml_pred == 0:  # ML predicts normal
            # Ensure we keep normal messages most of the time
            if np.random.random() < self.normal_keep_threshold:
                return 1  # force keep
            else:
                return raw_action  # use GRL action
        else:  # ML predicts malicious
            # Ensure we prune malicious messages most of the time
            if np.random.random() < self.malicious_prune_threshold:
                return 0  # force prune
            else:
                return raw_action  # use GRL action
    
    def update_error_rates(self, action, is_malicious):
        """Update error tracking for adaptive masking"""
        # Record decision outcome
        if action == 0 and is_malicious == 1:  # Correctly prune malicious
            self.recent_decisions["tp"] += 1
        elif action == 1 and is_malicious == 0:  # Correctly keep normal
            self.recent_decisions["tn"] += 1
        elif action == 0 and is_malicious == 0:  # FP - Wrongly prune normal
            self.recent_decisions["fp"] += 1
        elif action == 1 and is_malicious == 1:  # FN - Wrongly keep malicious
            self.recent_decisions["fn"] += 1
        
        # Calculate rates every 10 decisions
        if sum(self.recent_decisions.values()) >= 10:
            # Calculate FP rate
            normal_count = self.recent_decisions["tn"] + self.recent_decisions["fp"]
            if normal_count > 0:
                fp_rate = self.recent_decisions["fp"] / normal_count
                self.fp_rate_window.append(fp_rate)
                
            # Calculate FN rate
            malicious_count = self.recent_decisions["tp"] + self.recent_decisions["fn"]
            if malicious_count > 0:
                fn_rate = self.recent_decisions["fn"] / malicious_count
                self.fn_rate_window.append(fn_rate)
                
            # Keep window at specified size
            if len(self.fp_rate_window) > self.window_size:
                self.fp_rate_window.pop(0)
            if len(self.fn_rate_window) > self.window_size:
                self.fn_rate_window.pop(0)
                
            # Reset tracking
            self.recent_decisions = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    
    def evaluate_message(self, msg_data, ml_pred, ml_prob):
        """
        Evaluate a message using the loaded GRL model with action masking.
        Returns the GRL action (0=prune, 1=keep)
        """
        try:
            # Check cache first for previously seen similar messages
            cache_key = f"{msg_data.sender_id}_{msg_data.receiver_id}_{ml_prob:.2f}"
            if cache_key in self.action_cache:
                cached_action = self.action_cache[cache_key]
                # Still update error rates with cached action
                self.update_error_rates(cached_action, msg_data.is_malicious)
                return cached_action
            
            # Prepare observation for the model
            observation = self.prepare_observation(msg_data, ml_pred, ml_prob)
            
            # Get prediction from model
            action, _ = self.model.predict(observation, deterministic=True)
            
            # Convert action to integer (0=prune, 1=keep)
            raw_grl_action = int(action[0])
            
            # Apply action masking
            masked_grl_action = self.apply_action_masking(raw_grl_action, msg_data.is_malicious, ml_pred)
            
            # Update error rates
            self.update_error_rates(masked_grl_action, msg_data.is_malicious)
            
            # Cache the masked result
            self.action_cache[cache_key] = masked_grl_action
            
            # Log the message and decision
            with open(self.log_file, 'a') as f:
                line = (f"{time.time()},{msg_data.sender_id},{msg_data.receiver_id},"
                       f"{msg_data.pos_x},{msg_data.pos_y},{msg_data.spd_x},{msg_data.spd_y},"
                       f"{msg_data.acl_x},{msg_data.acl_y},{msg_data.hed_x},{msg_data.hed_y},"
                       f"{ml_pred},{ml_prob},{raw_grl_action},{masked_grl_action},{msg_data.is_malicious}\n")
                f.write(line)
            
            # Print diagnostic info
            if raw_grl_action != masked_grl_action:
                print(f"Action masking applied: raw GRL={raw_grl_action}, masked={masked_grl_action}")
                if len(self.fp_rate_window) > 0:
                    print(f"Current FP rate: {np.mean(self.fp_rate_window):.4f}, FN rate: {np.mean(self.fn_rate_window):.4f}")
                    print(f"Thresholds: normal_keep={self.normal_keep_threshold:.2f}, malicious_prune={self.malicious_prune_threshold:.2f}")
            
            return masked_grl_action
            
        except Exception as e:
            print(f"Error in GRL evaluation: {e}")
            traceback.print_exc()
            

def initialize_ml_model(checkpoint_path, args):
    """Initialize the ML model from checkpoint"""
    exp = Exp_Classification(args)
    exp.model = exp._build_model()
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_dict = exp.model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items()
                       if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    exp.model.load_state_dict(model_dict)
    exp.model.eval()
    return exp

def ml_prediction(exp, msg_data, processor):
    """Process data and make ML prediction"""
    try:
        # Process data using the processor
        normalized_values, label = processor.process_row(msg_data)
        
        # Convert to model input format
        batch_x = torch.FloatTensor(normalized_values).unsqueeze(0)
        padding_mask = torch.ones(1, 1).float()
        
        # Move to device
        batch_x = batch_x.to(exp.args.device)
        padding_mask = padding_mask.to(exp.args.device)
        
        # Model inference
        exp.model.eval()
        with torch.no_grad():
            outputs = exp.model(batch_x, padding_mask, None, None)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1).cpu().numpy()[0]
            prediction_prob = probs[0][1].item()  # Probability of being malicious
            
            print(f"ML Prediction: {prediction} (Malicious prob: {prediction_prob:.4f})")
        
        return prediction, prediction_prob
        
    except Exception as e:
        print(f"Error in ML processing: {e}")
        traceback.print_exc()
        return 0, 0.0

def handle_client(conn, ml_exp, processor, grl_evaluator):
    """Handle individual client connections"""
    try:
        data = b''
        remaining = 326  # Adjust based on MessageData structure size
        while remaining > 0:
            chunk = conn.recv(remaining)
            if not chunk:
                print("Connection closed by client")
                return
            data += chunk
            remaining -= len(chunk)
        
        print(f"Received data of size: {len(data)} bytes")
        
        msg_data = MessageData(data)
        
        # Make ML prediction 
        ml_pred, ml_prob = ml_prediction(ml_exp, msg_data, processor)
        
        # Run GRL evaluation 
        grl_action = grl_evaluator.evaluate_message(msg_data, ml_pred, ml_prob)
        
        should_prune = 1 if grl_action == 0 else 0
        
        # Pack both responses: [ML prediction, should_prune flag (1=prune, 0=keep)]
        response = struct.pack('ii', ml_pred, should_prune)
        conn.sendall(response)

        print(f"DECISION ANALYSIS:")
        print(f"True label from simulation: {msg_data.is_malicious} ({'Malicious' if msg_data.is_malicious == 1 else 'Normal'})")
        print(f"ML Prediction: {ml_pred} ({'Malicious' if ml_pred == 1 else 'Normal'}, Prob: {ml_prob:.4f})")
        print(f"GRL Action (w/ masking): {grl_action} ({'Keep' if grl_action == 1 else 'Prune'})")
        print(f"Final Decision: {'Prune' if should_prune == 1 else 'Keep'}")
            
    except Exception as e:
        print(f"Error handling client: {e}")
        traceback.print_exc()
    finally:
        conn.close()

def main():
    # Define ML model args
    args_dict = {
        "device": torch.device("cpu"),
        "use_multi_gpu": False,
        "use_gpu": False,
        "model": "Informer",
        "task_name": "classification",
        "data": "veremi",
        "model_id": "verClssSingle",
        "root_path": "C:/Users/Latifa/src/vanetTuto/simulations/veins_inet_openStreetMap/AD/data",
        "data_path": "simulation.csv",
        "features": "M",
        "target": "OT",
        "freq": "h",
        "checkpoints": "./checkpoints/",
        "seq_len": 1,
        "label_len": 0,
        "pred_len": 0,
        "enc_in": 8,
        "dec_in": 8,
        "c_out": 2,
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 2048,
        "moving_avg": 25,
        "factor": 1,
        "distil": True,
        "dropout": 0.1,
        "embed": "timeF",
        "activation": "gelu",
        "decomp_method": "moving_avg",
        "num_workers": 0,
        "itr": 1,
        "train_epochs": 7,
        "batch_size": 256,
        "patience": 5,
        "learning_rate": 0.0001,
        "des": "test",
        "loss": "MSE",
        "lradj": "type1",
        "use_amp": False,
        "p_hidden_dims": [128, 128],
        "p_hidden_layers": 2,
        "use_dtw": False,
        "top_k": 5,
        "num_class": 2,
    }
    args = Namespace(**args_dict)
    
    # Initialize data processor
    dataset_path = os.path.join(args.root_path, args.data_path)
    processor = DataProcessor(dataset_path, switch_threshold=50)
    
    # Initialize ML model
    print("Initializing ML model...")
    ml_checkpoint = 'C:/Users/Latifa/src/vanetTuto/simulations/veins_inet_openStreetMap/AD/checkpoints/checkpoints_finetuninginformer_best.pth'
    ml_exp = initialize_ml_model(ml_checkpoint, args)
    ml_exp.args = args
    print("ML model loaded successfully")
    
    # Initialize GRL evaluator with pre-trained model
    print("Initializing GRL evaluator...")
    grl_model_path = "C:/Users/Latifa/src/vanetTuto/simulations/veins_inet_openStreetMap/AD/GRL_Vanet_final_two/models/vanet_grl_model.zip"  
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grl_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    grl_evaluator = GRLEvaluator(grl_model_path, log_dir)
    print("GRL evaluator initialized")
    
    # Set up signal handlers for clean shutdown
    def signal_handler(sig, frame):
        print("Received signal to terminate. Shutting down...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start server
    HOST = '127.0.0.1'
    PORT = 5000
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(20)
        print(f"Server listening on {HOST}:{PORT}")
        
        while True:
            try:
                conn, addr = s.accept()
                print(f"Connected by {addr}")
                handle_client(conn, ml_exp, processor, grl_evaluator)
            except Exception as e:
                print(f"Error accepting connection: {e}")
                continue

if __name__ == "__main__":
    main()