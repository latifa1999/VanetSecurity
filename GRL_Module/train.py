import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt
from env import VANETGraphEnv
from gnn import CustomGNNPolicy
import json
from tqdm.auto import tqdm
import torch


class ProgressBarCallback(BaseCallback):
    """
    Custom callback to display a progress bar during training using tqdm.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_mean_reward = 0
        self.exploration_rate = 1.0  # Start with full exploration
        self.action_counts = {0: 0, 1: 0}  # Track action distribution
        self.recent_fp_fn = {'fp': [], 'fn': []}  # Track recent FP/FN
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")
    
    def _on_step(self):
        # Update progress bar based on the number of timesteps
        if self.pbar is not None:
            n_env = self.training_env.num_envs
            
            # Update reward info if available
            try:
                # Try to get rewards from environment directly
                if hasattr(self.training_env.envs[0], 'episode_rewards') and self.training_env.envs[0].episode_rewards:
                    self.last_mean_reward = np.mean(self.training_env.envs[0].episode_rewards[-10:]) if len(self.training_env.envs[0].episode_rewards) >= 10 else np.mean(self.training_env.envs[0].episode_rewards)
                # Try getting rewards from episode info buffer
                elif len(self.model.ep_info_buffer) > 0:
                    rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer if "r" in ep_info]
                    if rewards:
                        self.last_mean_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
            except Exception as e:
                # Just continue without updating reward
                pass
            
            # Calculate exploration rate (decaying over time)
            progress = self.num_timesteps / self.total_timesteps
            self.exploration_rate = max(0.05, 1.0 - progress)  # Minimum 5% exploration
            
            # Track action distribution from environment's action history if available
            try:
                if hasattr(self.training_env.envs[0], 'last_actions') and self.training_env.envs[0].last_actions is not None:
                    for action_idx, action in enumerate(self.training_env.envs[0].last_actions):
                        action_int = int(action)
                        if action_int in self.action_counts:
                            self.action_counts[action_int] += 1
                            
                # Track recent FP/FN rates
                if hasattr(self.training_env.envs[0], 'reward_debug'):
                    rd = self.training_env.envs[0].reward_debug
                    fp = rd.get('fp', 0)
                    fn = rd.get('fn', 0)
                    tp = rd.get('tp', 0)
                    tn = rd.get('tn', 0)
                    
                    # Calculate rates
                    fp_rate = fp / max(1, fp + tn) if (fp + tn) > 0 else 0
                    fn_rate = fn / max(1, fn + tp) if (fn + tp) > 0 else 0
                    
                    # Add to recent tracking
                    self.recent_fp_fn['fp'].append(fp_rate)
                    self.recent_fp_fn['fn'].append(fn_rate)
                    
                    # Keep only recent values
                    if len(self.recent_fp_fn['fp']) > 10:
                        self.recent_fp_fn['fp'].pop(0)
                    if len(self.recent_fp_fn['fn']) > 10:
                        self.recent_fp_fn['fn'].pop(0)
                    
            except Exception as e:
                pass
            
            # Update description with reward, exploration and FP/FN rates
            mean_fp = np.mean(self.recent_fp_fn['fp']) if self.recent_fp_fn['fp'] else 0
            mean_fn = np.mean(self.recent_fp_fn['fn']) if self.recent_fp_fn['fn'] else 0
            
            self.pbar.set_description(
                f"Training Progress | Reward: {self.last_mean_reward:.2f} | "
                f"FP: {mean_fp:.2f} | FN: {mean_fn:.2f} | "
                f"Actions: {self.action_counts[0]}/{self.action_counts[1]}"
            )
            
            # Update progress
            self.pbar.update(n_env)
        return True
        
    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
            
        # Print action distribution
        total_actions = sum(self.action_counts.values())
        print("\nAction Distribution Analysis:")
        if total_actions > 0:
            print(f"Action 0 (Prune/Remove Edge): {self.action_counts[0]} ({self.action_counts[0]/total_actions*100:.2f}%)")
            print(f"Action 1 (Keep Edge): {self.action_counts[1]} ({self.action_counts[1]/total_actions*100:.2f}%)")
            
            max_count = max(self.action_counts.values())
            if max_count > 0:
                min_count = min(self.action_counts.values())
                print(f"Action Balance Ratio: {min_count/max_count:.4f}")
            else:
                print("Action Balance Ratio: N/A (no actions recorded)")
        else:
            print("No actions were recorded during training.")


class TrainingMetricsCallback(BaseCallback):
    """
    Callback for saving training metrics to plot later.
    """
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.rewards = []
        self.ep_lengths = []
        self.losses = {"policy_loss": [], "value_loss": [], "entropy_loss": []}
        self.action_distribution = []
        self.actions_by_label = {
            0: {0: [], 1: []},  # For normal edges: tracking actions over time
            1: {0: [], 1: []}   # For malicious edges: tracking actions over time
        }
        self.timesteps = []
        self.collect_interval = 10  # Collect data every N steps to reduce volume
        self.step_counter = 0
        self.reward_breakdown = {
                "tp": [], "fp": [], "tn": [], "fn": []
        }
        self.fp_fn_rates = {
            "fp_rate": [],
            "fn_rate": []
        }

        
    def _on_step(self):
        # Only collect data at intervals to keep the amount manageable
        self.step_counter += 1
        if self.step_counter % self.collect_interval != 0:
            return True
            
        # Record current timestep
        self.timesteps.append(self.num_timesteps)
        
        # Log latest rewards
        try:
            # Try to get rewards from episode buffer
            if len(self.model.ep_info_buffer) > 0:
                recent_rewards = [float(ep_info["r"]) for ep_info in self.model.ep_info_buffer if "r" in ep_info]
                if recent_rewards:
                    # Use the latest reward
                    self.rewards.append(recent_rewards[-1])
                    
            # Try to get rewards from environment
            elif hasattr(self.training_env.envs[0], 'episode_rewards') and self.training_env.envs[0].episode_rewards:
                self.rewards.append(float(sum(self.training_env.envs[0].episode_rewards)))
                
            # If we added a reward, also add episode length
            if len(self.rewards) > len(self.ep_lengths):
                if len(self.model.ep_info_buffer) > 0 and "l" in self.model.ep_info_buffer[-1]:
                    self.ep_lengths.append(int(self.model.ep_info_buffer[-1]["l"]))
                else:
                    self.ep_lengths.append(0)  # Default if we can't get episode length
        except Exception as e:
            if self.verbose > 0:
                print(f"Error recording rewards: {e}")
        
        # Keep rewards and episode lengths lists the same length
        while len(self.rewards) > len(self.ep_lengths):
            self.ep_lengths.append(0)
        while len(self.ep_lengths) > len(self.rewards):
            self.rewards.append(0)
            
        # Log losses - modified to better capture loss data
        try:
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                all_logs = self.model.logger.name_to_value
                
                # Add explicit log keys to capture
                for loss_name in ['loss', 'policy_loss', 'value_loss', 'entropy_loss', 'approx_kl', 'clip_fraction']:
                    if loss_name in all_logs:
                        # Create the key if it doesn't exist
                        if loss_name not in self.losses:
                            self.losses[loss_name] = []
                        # Convert to standard float to ensure JSON serialization works
                        self.losses[loss_name].append(float(all_logs[loss_name]))
                        
                # Also try suffixed keys (e.g., 'train/loss')
                for full_key in all_logs.keys():
                    if '/' in full_key:
                        prefix, key = full_key.split('/', 1)
                        if key in ['loss', 'policy_loss', 'value_loss', 'entropy_loss']:
                            # Create the key if it doesn't exist
                            if key not in self.losses:
                                self.losses[key] = []
                            # Add to the list
                            self.losses[key].append(float(all_logs[full_key]))
                            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error recording losses: {e}")
        
        # Track action distribution
        try:
            env = self.training_env.envs[0]
            if hasattr(env, "reward_debug"):
                for key in ["tp", "fp", "tn", "fn"]:
                    self.reward_breakdown[key].append(env.reward_debug.get(key, 0))
                    
                # Calculate and track FP/FN rates
                tp = env.reward_debug.get("tp", 0)
                tn = env.reward_debug.get("tn", 0)
                fp = env.reward_debug.get("fp", 0)
                fn = env.reward_debug.get("fn", 0)
                
                normal_total = tn + fp
                malicious_total = tp + fn
                
                fp_rate = fp / max(1, normal_total) if normal_total > 0 else 0
                fn_rate = fn / max(1, malicious_total) if malicious_total > 0 else 0
                
                self.fp_fn_rates["fp_rate"].append(fp_rate)
                self.fp_fn_rates["fn_rate"].append(fn_rate)

            # Method 1: Use action_history if available
            if hasattr(env, 'action_history') and env.action_history:
                # Track counts at this timestep
                action_counts = {0: 0, 1: 0}
                label_action_counts = {
                    0: {0: 0, 1: 0},  # label 0: {action 0 count, action 1 count}
                    1: {0: 0, 1: 0}   # label 1: {action 0 count, action 1 count}
                }
                
                # Process all recorded actions
                for action_set in env.action_history:
                    for action, label in action_set:
                        action_int = int(action)
                        label_int = int(label)
                        
                        # Update overall action counts
                        if action_int in action_counts:
                            action_counts[action_int] += 1
                        
                        # Update label-specific action counts
                        if label_int in label_action_counts and action_int in label_action_counts[label_int]:
                            label_action_counts[label_int][action_int] += 1
                
                # Store the overall distribution
                self.action_distribution.append(action_counts)
                
                # Store label-specific distributions
                for label in [0, 1]:
                    for action in [0, 1]:
                        # Add this timestep's count
                        self.actions_by_label[label][action].append(label_action_counts[label][action])
        except Exception as e:
            if self.verbose > 0:
                print(f"Error recording actions: {e}")
            
            # Ensure we have default values in case of errors
            self.action_distribution.append({0: 0, 1: 0})
            for label in [0, 1]:
                for action in [0, 1]:
                    if len(self.actions_by_label[label][action]) < len(self.timesteps):
                        self.actions_by_label[label][action].append(0)
        
        return True
        
    def save_metrics(self, path):
        """Save metrics to file for later analysis"""
        metrics = {
            "rewards": self.rewards,
            "episode_lengths": self.ep_lengths,
            "timesteps": self.timesteps,
            "losses": self.losses,
            "action_distribution": self.action_distribution,
            "actions_by_label": self.actions_by_label,
            "reward_breakdown": self.reward_breakdown,
            "fp_fn_rates": self.fp_fn_rates
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save metrics as JSON
        with open(path, 'w') as f:
            json.dump(metrics, f)


class ActionTrackingCallback(BaseCallback):
    """
    Callback to track actions and the associated labels during training.
    This allows analysis of which actions are taken for each type of label.
    """
    def __init__(self, verbose=0):
        super(ActionTrackingCallback, self).__init__(verbose)
        # Dictionary to track action counts by true label
        self.actions_by_label = {
            0: {0: 0, 1: 0},  # label 0 (normal): {action 0 count, action 1 count}
            1: {0: 0, 1: 0}   # label 1 (malicious): {action 0 count, action 1 count}
        }
        
    def _on_step(self):
        try:
            # Try to get actions and labels directly from environment
            env = self.training_env.envs[0]
            
            # Method 1: Use explicitly stored last_actions and last_labels
            if hasattr(env, 'last_actions') and hasattr(env, 'last_labels') and env.last_actions is not None and env.last_labels is not None:
                for action, label in zip(env.last_actions, env.last_labels):
                    action_int = int(action)
                    label_int = int(label)
                    
                    # Track this action-label pair
                    if label_int in self.actions_by_label and action_int in self.actions_by_label[label_int]:
                        self.actions_by_label[label_int][action_int] += 1
            
            # Method 2: Use the action history
            elif hasattr(env, 'action_history') and env.action_history:
                # Get the latest set of action-label pairs
                latest_actions = env.action_history[-1]
                for action, label in latest_actions:
                    action_int = int(action)
                    label_int = int(label)
                    
                    # Track this action-label pair
                    if label_int in self.actions_by_label and action_int in self.actions_by_label[label_int]:
                        self.actions_by_label[label_int][action_int] += 1
                    
        except Exception as e:
            if self.verbose > 0:
                print(f"Error tracking actions: {e}")
        
        return True
    
    def get_action_distribution(self):
        return self.actions_by_label

class LearningRateScheduleCallback(BaseCallback):
    """
    Callback to implement a learning rate schedule during training.
    """
    def __init__(self, initial_lr=0.0005, min_lr=0.00005):
        super(LearningRateScheduleCallback, self).__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        
    def _on_step(self):
        # Calculate current progress fraction
        progress_fraction = self.num_timesteps / self.model.num_timesteps
        
        # Cosine annealing schedule
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress_fraction))
        
        # Set the new learning rate
        self.model.learning_rate = lr
        
        return True


def train_ppo_agent(env_path, model_save_path, metrics_save_path="logs/training_metrics.json", total_timesteps=100000):
    """
    Train a PPO agent on the VANET environment.
    
    Args:
        env_path: Path to the data file
        model_save_path: Where to save the trained model
        metrics_save_path: Where to save training metrics
        total_timesteps: Number of steps to train for
    """
    # Create the environment
    env = VANETGraphEnv(env_path)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # Best hyperparameters from optimization
    hyperparams = {
        'learning_rate': 0.0003,        # Lower learning rate for more stability
        'n_steps': 2048,                # Larger batch of experiences before update
        'batch_size': 256,              # Larger batch size for better gradient estimates
        'n_epochs': 8,                  # More optimization epochs per update
        'gamma': 0.995,                 # High discount factor for long-term rewards
        'gae_lambda': 0.95,             # Slightly reduced from previous for better advantage estimation
        'clip_range': 0.2,              # Standard clipping for PPO
        'clip_range_vf': 0.3,           # Value function clipping for stability
        'ent_coef': 0.005,               # Moderate entropy for exploration
        'vf_coef': 0.7,                 # Higher value function coefficient for stability
        'max_grad_norm': 0.7,           # Slightly higher gradient clipping for better learning
        'use_sde': True,                # Enable State Dependent Exploration
        'sde_sample_freq': 4            # Sample SDE every 4 steps
    }

    
    # Initialize PPO agent with custom policy and improved hyperparameters
    policy_kwargs = {
        'features_extractor_class': CustomGNNPolicy,
        'features_extractor_kwargs': {
            'features_dim': 512  # Increased feature dimensions for better representation
        },
        'net_arch': [
            dict(
                pi=[512, 384, 256],  # Wider and deeper policy network
                vf=[512, 384, 256]   # Matching value function network
            )
        ],
        'activation_fn': torch.nn.GELU,  # GELU activation often works better than ReLU
        'optimizer_kwargs': {
            'eps': 1e-7,             # Small epsilon for numerical stability
            'weight_decay': 0.01     # L2 regularization to prevent overfitting
        }
    }
    
    # Create model with better hyperparameters for this task
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/tensorboard/",
        **hyperparams
    )
    
    # Create callbacks for tracking metrics and progress
    metrics_callback = TrainingMetricsCallback(verbose=1)  # Added verbosity
    progress_callback = ProgressBarCallback(total_timesteps=total_timesteps)
    action_callback = ActionTrackingCallback(verbose=1)    # Added verbosity
    lr_callback = LearningRateScheduleCallback(
        initial_lr=hyperparams['learning_rate'],
        min_lr=hyperparams['learning_rate'] * 0.1
    )
    
    # Create checkpoint callback - save every 10000 steps
    checkpoint_dir = os.path.join(os.path.dirname(model_save_path), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="vanet_grl_model_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )

    # Create eval environment for periodic evaluation
    eval_env = VANETGraphEnv(env_path)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # Sync normalization statistics from training to eval env
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    
    # Add evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(os.path.dirname(model_save_path), "best_model"),
        log_path="./logs/eval/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Combine callbacks
    all_callbacks = [metrics_callback, progress_callback, action_callback, checkpoint_callback, lr_callback, eval_callback]
    
    # Train the agent
    model.learn(total_timesteps=total_timesteps, callback=all_callbacks)
    
    # Get action distribution by label
    actions_by_label = action_callback.get_action_distribution()
    print("\nAction Distribution by True Label:")
    
    try:
        for label in [0, 1]:
            if label in actions_by_label:
                label_name = "Normal" if label == 0 else "Malicious"
                total = sum(actions_by_label[label].values())
                
                print(f"{label_name} Edges:")
                if total > 0:
                    action0_pct = actions_by_label[label][0] / total * 100
                    action1_pct = actions_by_label[label][1] / total * 100
                    
                    print(f"  Action 0 (Prune): {actions_by_label[label][0]} ({action0_pct:.2f}%)")
                    print(f"  Action 1 (Keep): {actions_by_label[label][1]} ({action1_pct:.2f}%)")
                    
                    # For malicious, we want high prune rate (Action 0)
                    # For normal, we want high keep rate (Action 1)
                    if label == 0:
                        correctness = action1_pct
                    else:
                        correctness = action0_pct
                        
                    print(f"  Correct Action Rate: {correctness:.2f}%")
                    
                    # Report the balance ratio between actions
                    if label == 0 and action1_pct < 50:
                        print(f"  WARNING: Model is biased toward pruning normal edges.")
                    elif label == 1 and action0_pct < 50:
                        print(f"  WARNING: Model is biased toward keeping malicious edges.")
                else:
                    print("  No actions recorded for this label type")
    except Exception as e:
        print(f"Error calculating action distribution by label: {e}")
    
    # Save metrics including action distribution
    metrics_callback.save_metrics(metrics_save_path)
    
    # Save the trained model without extension
    # The .zip is automatically added by stable-baselines3
    model_path = model_save_path.replace('.zip', '')
    model.save(model_path)
    print(f"Final model saved to {model_path} (StableBaselines3 will add .zip extension)")
    
    return model


if __name__ == "__main__":
    # File paths
    data_path = "data.csv" 
    model_save_path = "models/vanet_grl_model"
    
    # Train the model
    print("Training PPO agent...")
    model = train_ppo_agent(data_path, model_save_path, total_timesteps=100000)
    
    print("\nTraining complete!")