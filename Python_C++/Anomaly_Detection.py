#!/usr/bin/env python3

import os
import socket
import struct
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
from sklearn.metrics import f1_score, accuracy_score
from exp.exp_classification import Exp_Classification
from collections import deque

class MessageData:
    def __init__(self, data):
        try:
            current_pos = 0
            
            # 1) sender_id (100 bytes)
            self.sender_id = data[current_pos:current_pos+100].split(b'\0')[0].decode('utf-8', errors='ignore')
            current_pos += 100
            print(f"Sender ID: {self.sender_id}")
            
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
            self.attack_type = data[current_pos:current_pos+42].split(b'\0')[0].decode('utf-8', errors='ignore')
            current_pos += 42
            
            # 5) is_malicious (4 bytes)
            malicious_data = data[current_pos:current_pos+4]
            if len(malicious_data) == 4:
                self.is_malicious = struct.unpack('i', malicious_data)[0]
            else:
                self.is_malicious = 0
                
        except Exception as e:
            print(f"Error unpacking data: {e}")
            raise

class OnlineDataProcessor:
    def __init__(self, initial_dataset_path, switch_threshold=100):
        self.switch_threshold = switch_threshold  # Number of samples before switching
        self.collected_data = []  # Store simulation data
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
        
        print("\nFinetuning data statistics:")
        for col in self.feature_columns:
            print(f"{col}:")
            print(f"  Range: [{self.ranges[col][0]:.4f}, {self.ranges[col][1]:.4f}]")
            print(f"  Mean: {self.finetune_means[col]:.4f}")
            print(f"  Std: {self.finetune_stds[col]:.4f}")
    
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
            print("\nWARNING: Zero standard deviation in simulation data for features:", 
                  stds[zero_stds].index.tolist())
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
        
        # Clip values to ranges from finetuning data
        for col in self.feature_columns:
            min_val, max_val = self.ranges[col]
            if features[col].iloc[0] < min_val or features[col].iloc[0] > max_val:
                print(f"\nWARNING: Clipping {col} from {features[col].iloc[0]:.4f} to [{min_val:.4f}, {max_val:.4f}]")
            features[col] = features[col].clip(min_val, max_val)
        
        # Choose normalization statistics
        sim_means, sim_stds = self.get_simulation_statistics()
        if sim_means is not None:
            #print(f"\nUsing simulation statistics (based on {len(self.collected_data)} samples)")
            means, stds = sim_means, sim_stds
        else:
            #print("\nUsing finetuning statistics")
            means, stds = self.finetune_means, self.finetune_stds
        
        # Normalize features
        normalized = (features - means) / stds
        
        return normalized.values, label

def initialize_model(checkpoint_path, args):
    """Initialize the model from checkpoint"""
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

def process_and_predict(exp, msg_data, processor):
    """Process data and make prediction"""
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
            
            print("\nModel outputs:")
            #print(f"Raw outputs: {outputs}")
            print(f"Probabilities: {probs}")
            print(f"Prediction: {prediction}")
            print(f"True label: {label[0]}")
        
        return prediction
        
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return 0

def handle_client(conn, exp, processor):
    """Handle individual client connections"""
    try:
        data = b''
        remaining = 326
        while remaining > 0:
            chunk = conn.recv(remaining)
            if not chunk:
                print("Connection closed by client")
                return
            data += chunk
            remaining -= len(chunk)
        
        msg_data = MessageData(data)
        prediction = process_and_predict(exp, msg_data, processor)

        # pack response
        response = struct.pack('i', prediction)
        conn.sendall(response)

        # Print detailed status
        print(f"\nProcessed message from {msg_data.sender_id}:")
        print(f"Prediction: {'Malicious' if prediction == 1 else 'Normal'}")
        print(f"True Label: {msg_data.is_malicious}")
            
    except Exception as e:
        print(f"Error handling client: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

def main():
    # Define args
    args_dict = {
        "device": torch.device("cpu"),
        "use_multi_gpu": False,
        "use_gpu": False,
        "model": "Informer",
        "task_name": "classification",
        "data": "veremi",
        "model_id": "verClssSingle",
        "root_path": "C:/Users/Latifa/src/vanetTuto/simulations/veins_inet_openStreetMap/AD/data/Normalization",
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
    processor = OnlineDataProcessor(dataset_path, switch_threshold=50)
    
    # Initialize model
    print("\nInitializing model...")
    cp = 'C:/Users/Latifa/src/vanetTuto/simulations/veins_inet_openStreetMap/AD/checkpoints/checkpoints_finetuninginformer_best.pth'
    exp = initialize_model(cp, args)
    exp.args = args
    print("Model loaded successfully")

    # Start server
    HOST = '127.0.0.1'
    PORT = 5000
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(20)
        print(f"\nServer listening on {HOST}:{PORT}")
        
        while True:
            try:
                conn, addr = s.accept()
                print(f"\nConnected by {addr}")
                handle_client(conn, exp, processor)
            except Exception as e:
                print(f"Error accepting connection: {e}")
                continue

if __name__ == "__main__":
    main()