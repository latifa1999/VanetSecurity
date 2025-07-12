import os
import argparse
from train import train_ppo_agent
from evaluate import evaluate_model, compare_ml_vs_grl
from visualize import plot_training_metrics, plot_confusion_matrices, plot_comparative_metrics, plot_action_distribution, plot_improved_training_rewards, plot_rule_based_comparison, plot_reward_breakdown, plot_smooth_training_rewards
from stable_baselines3 import PPO
import json
import numpy as np

import time
from datetime import datetime

# Import our monitoring utilities
from utils import count_parameters, get_memory_usage, get_gpu_usage, TrainingMonitor

def main(args):
    """
    Main function to run the entire pipeline or specific components.P
    """
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # File paths
    #data_path = args.data_path
    train_path = args.train_path
    eval_path = args.eval_path
    model_base_path = "models/vanet_grl_model"  # Base path without extension
    metrics_path = "logs/training_metrics.json"
    results_path = "logs/evaluation_results.json"

    # Create a timestamp for monitoring logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    monitor_log_path = f"training_monitor_{timestamp}.json"
    
    # Create the training monitor
    monitor = TrainingMonitor(log_interval=args.monitor_interval)
    
    # Run the appropriate step(s)
    if args.train:
        print("=== Training PPO Agent ===")
        print(f"Starting training with {args.timesteps} timesteps...")

        # Get initial resource usage
        print("\nInitial resource usage:")
        ram_usage = get_memory_usage()
        print(f"RAM: {ram_usage['memory_usage_mb']:.2f} MB ({ram_usage['memory_percent']:.2f}%)")
        
        gpu_usage = get_gpu_usage()
        if gpu_usage:
            for gpu in gpu_usage:
                print(f"GPU {gpu['device_id']} ({gpu['name']}): "
                      f"{gpu['memory_allocated_mb']:.2f} MB / {gpu['max_memory_mb']:.2f} MB "
                      f"({gpu['utilization_percent']:.2f}%)")
        else:
            # get CPU usage if no GPU is available
            print("No GPU available. Monitoring CPU usage instead.")
            cpu_usage = get_memory_usage()
            print(f"CPU: {cpu_usage['memory_usage_mb']:.2f} MB ({cpu_usage['memory_percent']:.2f}%)")
        
        # Start monitoring and record time
        monitor.start()
        
        model = train_ppo_agent(
            train_path, 
            model_base_path, 
            metrics_path, 
            total_timesteps=args.timesteps,
        )

        monitor.end()

        num_params = count_parameters(model)
        print(f"\nModel has {num_params: ,} trainable parameters.")

        # Print training summary
        monitor.print_summary()
        
        # Save monitoring logs
        monitor.save_logs(monitor_log_path)
        print(f"Monitoring logs saved to {monitor_log_path}")

        print("Training complete!")
    
    if args.evaluate:
        print("\n=== Evaluating Model ===")
        if 'model' not in locals():
            try:
                # Try loading with .zip extension first (stable-baselines3 adds it automatically)
                if os.path.exists(f"{model_base_path}.zip"):
                    print(f"Loading model from {model_base_path}.zip")
                    model = PPO.load(f"{model_base_path}.zip")
                # Try without extension
                elif os.path.exists(model_base_path):
                    print(f"Loading model from {model_base_path}")
                    model = PPO.load(model_base_path)
                else:
                    # Try loading latest checkpoint
                    checkpoint_dir = os.path.join(os.path.dirname(model_base_path), "checkpoints")
                    if os.path.exists(checkpoint_dir):
                        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                        if checkpoints:
                            # Sort checkpoints by step number
                            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else 0)
                            latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
                            print(f"Loading latest checkpoint: {latest_checkpoint}")
                            model = PPO.load(latest_checkpoint)
                        else:
                            print("No checkpoints found.")
                            return
                    else:
                        print("No model or checkpoint directory found.")
                        return
                
                print(f"Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                return
            
        # Evaluate the model
        metrics = evaluate_model(model, eval_path, num_episodes=args.eval_episodes)
        
        # Compare ML vs GRL
        comparison = compare_ml_vs_grl(metrics)
        comparison = convert_numpy_types(comparison)

        # Add tracking of correct percentages
        true_labels = np.array(metrics['true_labels'])
        grl_actions = np.array([1-p for p in metrics['grl_predictions']])  # Convert back to actions
        
        from evaluate import track_correct_actions
        correct_actions = track_correct_actions(true_labels, metrics['ml_predictions'], grl_actions)
        print("\nCorrect Action Percentages:")
        print(f"  Normal edges: {correct_actions['normal_correct_pct']:.2f}% ({correct_actions['normal_correct']}/{correct_actions['normal_total']})")
        print(f"  Malicious edges: {correct_actions['malicious_correct_pct']:.2f}% ({correct_actions['malicious_correct']}/{correct_actions['malicious_total']})")
        print(f"  Overall: {correct_actions['overall_correct_pct']:.2f}%")
        
        # Save evaluation results
        with open(results_path, 'w') as f:
            json.dump(comparison, f, indent=4)
        
        # Print results
        print("\nPerformance Comparison:")
        print("\nML Model:")
        for k, v in comparison['ML'].items():
            if k != 'Confusion Matrix':
                print(f"  {k}: {v}")
        
        print("\nGRL Model:")
        for k, v in comparison['GRL'].items():
            if k != 'Confusion Matrix':
                print(f"  {k}: {v}")
        
        print("\nImprovements:")
        for k, v in comparison['GRL_vs_Rule_Based'].items():
            print(f"  {k}: {v:.2f}%")

        print("\nadjusted Improvements:")
        for k, v in comparison['GRL_adjusted_vs_ML'].items():
            print(f"  {k}: {v:.2f}%")
        
        print("Evaluation complete!")
    
    if args.visualize:
        print("\n=== Generating Visualizations ===")
        
        # Plot training metrics if file exists
        if os.path.exists(metrics_path):
            print("Plotting training metrics...")
            plot_training_metrics(metrics_path)
            plot_improved_training_rewards(metrics_path)
        else:
            print(f"Training metrics file {metrics_path} not found. Skipping training plots.")
        
        # Plot evaluation results if file exists
        if os.path.exists(results_path):
            print("Plotting evaluation results...")
            with open(results_path, 'r') as f:
                comparison = json.load(f)
                
            # Convert lists back to numpy arrays
            for model_type in ['ML', 'GRL']:
                if 'Confusion Matrix' in comparison[model_type]:
                    comparison[model_type]['Confusion Matrix'] = np.array(
                        comparison[model_type]['Confusion Matrix']
                    )
            
            plot_confusion_matrices(comparison)
            plot_comparative_metrics(comparison)
            plot_action_distribution(metrics_path) 
            plot_rule_based_comparison(comparison)
            with open(metrics_path, 'r') as f:
                training_metrics = json.load(f)
            plot_reward_breakdown(training_metrics)
            plot_smooth_training_rewards(training_metrics)



        else:
            print(f"Evaluation results file {results_path} not found. Using sample data for visualization...")
            # Use the sample comparison from visualize.py
            import visualize
            visualize.main()
        
        print("Visualization complete! Check the 'plots' directory for output images.")

def convert_numpy_types(obj):
    """
    Convert NumPy types to standard Python types to ensure JSON serialization works.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VANET GRL Pipeline')
    parser.add_argument('--data_path', type=str, default='data.csv', 
                        help='Path to the VANET dataset CSV file')
    parser.add_argument('--train', action='store_true', 
                        help='Run the training step')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Run the evaluation step')
    parser.add_argument('--visualize', action='store_true', 
                        help='Generate visualizations')
    parser.add_argument('--all', action='store_true', 
                        help='Run all steps (train, evaluate, visualize)')
    parser.add_argument('--timesteps', type=int, default=100000, 
                        help='Number of timesteps for training')
    parser.add_argument('--eval_episodes', type=int, default=10, 
                        help='Number of episodes for evaluation')
    parser.add_argument('--eval_path', type=str, default = 'eval_data.csv',
                    help='Path to the evaluation dataset CSV file')
    parser.add_argument('--train_path', type=str, default = 'train_data.csv',
                    help='Path to the training dataset CSV file')
    parser.add_argument('--monitor_training', action='store_true',
                    help='Monitor resource usage during training')
    parser.add_argument('--monitor_interval', type=int, default=10000,
                    help='Interval (in timesteps) for logging resource usage during training')
    
    args = parser.parse_args()
    
    # If --all is specified, run everything
    if args.all:
        args.train = args.evaluate = args.visualize  = args.monitor_training = True
    
    # If no specific steps are specified, run everything
    if not any([args.train, args.evaluate, args.visualize]):
        args.train = args.evaluate = args.visualize = args.monitor_training = True
    
    main(args)