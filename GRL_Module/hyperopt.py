import os
import optuna
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from env import VANETGraphEnv
from gnn import CustomGNNPolicy
from tqdm.auto import tqdm

# Set up logging for Optuna
import logging
optuna.logging.get_logger("optuna").setLevel(logging.INFO)

def create_env(data_path):
    """Create a vectorized environment for training and evaluation"""
    env = VANETGraphEnv(data_path)
    env = DummyVecEnv([lambda: env])
    return env

def evaluate_hyperparams(model, env, n_eval_episodes=10):
    """Evaluate a model on metrics that matter for this specific task"""
    # Collect actions and labels over multiple episodes
    all_actions = []
    all_labels = []
    all_rewards = []
    
    for _ in range(n_eval_episodes):
        obs, _ = env.envs[0].reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Map actions to predictions (0=prune=malicious, 1=keep=normal)
            predictions = 1 - action  # Invert because 0=malicious, 1=normal in our labels
            
            # Store for later analysis
            all_actions.extend(action)
            all_labels.extend(env.envs[0].current_labels[:len(action)])
            
            # Take action in environment
            obs, reward, done, _, _ = env.envs[0].step(action)
            episode_reward += reward
        
        all_rewards.append(episode_reward)
    
    # Convert to arrays for analysis
    actions = np.array(all_actions).flatten()
    labels = np.array(all_labels).flatten()
    
    # Calculate class-specific metrics
    normal_indices = np.where(labels == 0)[0]
    malicious_indices = np.where(labels == 1)[0]
    
    # For normal edges, action 1 (keep) is correct
    normal_correct = np.sum(actions[normal_indices] == 1) if len(normal_indices) > 0 else 0
    normal_total = len(normal_indices)
    normal_accuracy = normal_correct / normal_total if normal_total > 0 else 0
    
    # For malicious edges, action 0 (prune) is correct
    malicious_correct = np.sum(actions[malicious_indices] == 0) if len(malicious_indices) > 0 else 0
    malicious_total = len(malicious_indices)
    malicious_accuracy = malicious_correct / malicious_total if malicious_total > 0 else 0
    
    # Combined metrics
    balanced_accuracy = (normal_accuracy + malicious_accuracy) / 2
    overall_accuracy = (normal_correct + malicious_correct) / (normal_total + malicious_total) if (normal_total + malicious_total) > 0 else 0
    avg_reward = np.mean(all_rewards)
    
    # Custom score that balances accuracy for both classes
    # We want to particularly emphasize keeping normal edges correctly
    custom_score = (normal_accuracy * 2 + malicious_accuracy) / 3
    
    return {
        'normal_accuracy': normal_accuracy,
        'malicious_accuracy': malicious_accuracy,
        'balanced_accuracy': balanced_accuracy,
        'overall_accuracy': overall_accuracy,
        'avg_reward': avg_reward,
        'custom_score': custom_score
    }

class TrialProgressBar:
    """
    Custom progress bar for tracking Optuna trials
    """
    def __init__(self, n_trials):
        self.pbar = tqdm(total=n_trials, desc="Hyperparameter Optimization")
        self.trial_count = 0
        
    def update(self, study, trial):
        """Update progress bar after each trial"""
        self.trial_count += 1
        self.pbar.update(1)
        
        # Display best score and current trial's score
        curr_score = trial.value if trial.value is not None else float('-inf')
        best_score = study.best_value if study.best_value is not None else float('-inf')
        
        self.pbar.set_description(
            f"Hyperparameter Optimization | Best: {best_score:.4f} | Current: {curr_score:.4f}"
        )
    
    def close(self):
        """Close the progress bar"""
        self.pbar.close()

class TrainingProgressBarCallback(BaseCallback):
    """
    Custom callback to track training progress with a tqdm progress bar.
    """
    def __init__(self, pbar, verbose=0):
        super(TrainingProgressBarCallback, self).__init__(verbose)
        self.pbar = pbar
        self.last_mean_reward = 0
        self.n_calls = 0
        
    def _on_step(self):
        self.n_calls += 1
        
        # Update reward info if available
        if len(self.model.ep_info_buffer) > 0:
            self.last_mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer if "r" in ep_info])
            
            # Update the progress bar description with rewards
            if self.n_calls % 10 == 0:  # Update description less frequently to avoid flickering
                self.pbar.set_description(
                    f"Training | Reward: {self.last_mean_reward:.2f}"
                )
        
        # Update progress
        self.pbar.update(self.training_env.num_envs)
        return True
        
    def _on_training_end(self):
        # Make sure the progress bar is complete
        if self.pbar.n < self.pbar.total:
            self.pbar.update(self.pbar.total - self.pbar.n)
        self.pbar.set_description(
            f"Training Complete | Final Reward: {self.last_mean_reward:.2f}"
        )

def optimize_hyperparam(trial, data_path, n_timesteps=100000, progress_bar=None):
    """Optuna objective function for hyperparameter optimization"""
    # Environment setup
    env = create_env(data_path)
    eval_env = create_env(data_path)
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 32, 2048, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, log=True)
    n_epochs = trial.suggest_int("n_epochs", 5, 50)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    features_dim = trial.suggest_categorical("features_dim", [128, 256, 512])
    
    # Create a progress bar for this trial
    train_pbar = tqdm(total=n_timesteps, desc=f"Trial {trial.number+1}", leave=False)
    
    # Architecture
    policy_kwargs = {
        'features_extractor_class': CustomGNNPolicy,
        'features_extractor_kwargs': {'features_dim': features_dim},
        'net_arch': [dict(pi=[256, 256], vf=[256, 256])]  # Deeper network
    }
    
    # Create model with sampled hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=None
    )
    
    # Create the progress bar callback
    progress_callback = TrainingProgressBarCallback(train_pbar)
    
    # Train the model
    try:
        model.learn(total_timesteps=n_timesteps, callback=progress_callback)
        
        # Evaluate the model
        metrics = evaluate_hyperparams(model, eval_env)
        
        # Log intermediate metrics
        trial.report(metrics['custom_score'], step=n_timesteps)
        
        # Close the progress bar
        train_pbar.close()
        
        # Clean up to prevent memory issues
        del model
        
        return metrics['custom_score']
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        train_pbar.close()
        return float('-inf')

def run_hyperparameter_optimization(data_path, n_trials=50, n_timesteps_per_trial=100000):
    """Run hyperparameter optimization using Optuna"""
    study_name = "vanet_grl_optimization"
    storage_name = f"sqlite:///{study_name}.db"
    
    # Create study or load existing one
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
    except:
        # If database doesn't exist, create a new study without storage
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
    
    # Create progress bar for tracking trials
    progress_callback = TrialProgressBar(n_trials)
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: optimize_hyperparam(trial, data_path, n_timesteps_per_trial, progress_callback),
            n_trials=n_trials,
            callbacks=[progress_callback.update],
            show_progress_bar=False  # We use our custom progress bar
        )
    finally:
        # Close progress bar in case of exception
        progress_callback.close()
    
    # Print results
    print("\nOptimization Complete!")
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)
    
    # Print distributions of hyperparameters
    print("\nHyperparameter Importance:")
    importance = optuna.importance.get_param_importances(study)
    for param, score in importance.items():
        print(f"  {param}: {score:.4f}")
    
    # Save best hyperparameters
    os.makedirs("logs", exist_ok=True)
    pd.DataFrame(study.best_params, index=[0]).to_csv("logs/best_hyperparameters.csv", index=False)
    
    return study.best_params

def train_with_best_hyperparameters(data_path, model_save_path, hyperparams, total_timesteps=50000):
    """Train a model with the best hyperparameters found"""
    # Create environment
    env = create_env(data_path)
    
    # Extract hyperparameters
    features_dim = hyperparams.pop('features_dim', 256)
    
    # Create policy_kwargs
    policy_kwargs = {
        'features_extractor_class': CustomGNNPolicy,
        'features_extractor_kwargs': {'features_dim': features_dim},
        'net_arch': [dict(pi=[256, 256], vf=[256, 256])]
    }
    
    # Create model with best hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,  # Set to 0 to avoid cluttering the console output
        tensorboard_log="./logs/tensorboard/",
        **hyperparams
    )
    
    # Create a progress bar
    pbar = tqdm(total=total_timesteps, desc="Training with Best Hyperparameters")
    
    # Create the progress bar callback
    progress_callback = TrainingProgressBarCallback(pbar)
    
    # Train the model
    try:
        model.learn(total_timesteps=total_timesteps, callback=progress_callback)
    finally:
        pbar.close()
    
    # Save the model
    model_path = model_save_path.replace('.zip', '')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    data_path = "data.csv"
    model_save_path = "models/vanet_grl_optimized"
    
    # Run hyperparameter optimization
    print("Running hyperparameter optimization...")
    n_trials = 50 
    n_timesteps_per_trial = 100000  
    
    best_params = run_hyperparameter_optimization(
        data_path, 
        n_trials=n_trials, 
        n_timesteps_per_trial=n_timesteps_per_trial
    )
    
    # Ask user if they want to train with best hyperparameters
    train_best = input("\nTrain model with best hyperparameters? (y/n): ")
    if train_best.lower() == 'y':
        print("\nTraining model with best hyperparameters...")
        total_timesteps = int(input("Enter number of timesteps for training (default: 50000): ") or "50000")
        train_with_best_hyperparameters(data_path, model_save_path, best_params, total_timesteps=total_timesteps)