import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np


def plot_training_metrics(metrics_path, save_dir="plots"):
    """
    Plot training metrics from the saved JSON file.
    
    Args:
        metrics_path: Path to the JSON file with training metrics
        save_dir: Directory to save the plots
    """
    # Load the metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if we have valid reward data
    if "rewards" not in metrics or not metrics["rewards"]:
        print("No valid reward data found. Creating placeholder plot instead.")
        plt.figure(figsize=(10, 6))
        plt.title('Episode Rewards During Training (No Data Available)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.text(0.5, 0.5, 'No reward data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_rewards.png")
        plt.close()
        return
    
    # Filter out any zero rewards which might be placeholder values
    rewards = [r for r in metrics['rewards'] if r != 0]
    
    if not rewards:
        print("No non-zero reward data found. Creating placeholder plot instead.")
        plt.figure(figsize=(10, 6))
        plt.title('Episode Rewards During Training (No Valid Data)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.text(0.5, 0.5, 'No valid reward data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_rewards.png")
        plt.close()
    else:
        # Plot episode rewards
        plt.figure(figsize=(10, 6))
        normalized_rewards = [r / (l if l > 0 else 1) for r, l in zip(metrics['rewards'], metrics['episode_lengths'])]
        plt.plot(normalized_rewards)
        #plt.plot(rewards)
        plt.title('Episode Normlized Rewards During Training')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_rewards.png")
        plt.close()
        
        # Plot moving average of rewards for smoother visualization
        if len(rewards) > 1:
            plt.figure(figsize=(10, 6))
            window_size = min(20, len(rewards))
            rewards_series = pd.Series(rewards)
            rolling_mean = rewards_series.rolling(window=window_size).mean()
            plt.plot(rolling_mean, color='blue', linewidth=2)
            plt.plot(rewards_series, color='lightgray', alpha=0.3)
            plt.title(f'Episode Rewards Moving Average (window={window_size})')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/training_rewards_moving_avg.png")
            plt.close()
    
    # Check if we have valid episode length data
    if "episode_lengths" in metrics and any(metrics["episode_lengths"]):
        # Filter out any zeros
        ep_lengths = [l for l in metrics['episode_lengths'] if l > 0]
        if ep_lengths:
            plt.figure(figsize=(10, 6))
            plt.plot(ep_lengths)
            plt.title('Episode Lengths During Training')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/episode_lengths.png")
            plt.close()
    
    # Check if we have valid loss data
    if "losses" in metrics:
        has_loss_data = False
        plt.figure(figsize=(10, 6))
        for loss_name, loss_values in metrics['losses'].items():
            if loss_values:  # Only plot if we have values
                has_loss_data = True
                plt.plot(loss_values, label=loss_name)
        
        if has_loss_data:
            plt.title('Training Losses')
            plt.xlabel('Update')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/training_losses.png")
        else:
            plt.title('Training Losses (No Data Available)')
            plt.xlabel('Update')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.text(0.5, 0.5, 'No loss data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/training_losses.png")
        plt.close()

def plot_improved_training_rewards(metrics_path, save_dir="plots"):
    """
    Create improved visualizations for training rewards with smoothing.
    
    Args:
        metrics_path: Path to the JSON file with training metrics
        save_dir: Directory to save the plots
    """
    # Load the metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if we have valid reward data
    if "rewards" not in metrics or not metrics["rewards"]:
        print("No valid reward data found.")
        return
    
    # Filter out any zero rewards which might be placeholder values
    rewards = [r for r in metrics['rewards'] if r != 0]
    
    if not rewards:
        print("No non-zero reward data found.")
        return
    
    # Plot episode rewards with smoothing
    plt.figure(figsize=(12, 7))
    
    # Plot raw rewards as light points
    plt.plot(rewards, 'o', alpha=0.3, color='lightblue', label='Raw rewards')
    
    # Apply different smoothing windows
    window_sizes = [5, 20, 50]
    colors = ['#3498db', '#2980b9', '#1f618d']
    
    for i, window in enumerate(window_sizes):
        if len(rewards) > window:
            smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
            plt.plot(smoothed, linewidth=2, color=colors[i], 
                     label=f'Moving average (window={window})')
    
    # Add episode markers at regular intervals
    interval = max(1, len(rewards) // 10)
    for i in range(0, len(rewards), interval):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    std_reward = np.std(rewards)
    
    # Add statistics annotation
    stats_text = (
        f"Mean: {mean_reward:.1f}\n"
        f"Median: {median_reward:.1f}\n"
        f"Std Dev: {std_reward:.1f}\n"
        f"Min: {min(rewards):.1f}\n"
        f"Max: {max(rewards):.1f}"
    )
    
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Enhance the plot
    plt.title('Improved Training Rewards Visualization', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/improved_training_rewards.png", dpi=300)
    plt.close()
    
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(rewards, kde=True, bins=30)
    plt.axvline(mean_reward, color='r', linestyle='--', 
                label=f'Mean: {mean_reward:.1f}')
    plt.axvline(median_reward, color='g', linestyle=':', 
                label=f'Median: {median_reward:.1f}')
    
    plt.title('Distribution of Episode Rewards', fontsize=16)
    plt.xlabel('Reward', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reward_distribution.png", dpi=300)
    plt.close()


def plot_confusion_matrices(comparison, save_dir="plots"):
    """
    Plot confusion matrices for ML and GRL models.
    
    Args:
        comparison: Dict with comparison metrics
        save_dir: Directory to save the plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get confusion matrices
    ml_cm = np.array(comparison['ML']['Confusion Matrix']) if isinstance(comparison['ML']['Confusion Matrix'], list) else comparison['ML']['Confusion Matrix']
    grl_cm = np.array(comparison['GRL']['Confusion Matrix']) if isinstance(comparison['GRL']['Confusion Matrix'], list) else comparison['GRL']['Confusion Matrix']
    
    # Plot ML confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=ml_cm, display_labels=['Normal', 'Malicious'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('ML Model Confusion Matrix')
    plt.savefig(f"{save_dir}/ml_confusion_matrix.png")
    plt.close()
    
    # Plot GRL confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=grl_cm, display_labels=['Normal', 'Malicious'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('GRL Model Confusion Matrix')
    plt.savefig(f"{save_dir}/grl_confusion_matrix.png")
    plt.close()

        # Combined confusion matrices (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ConfusionMatrixDisplay(confusion_matrix=ml_cm, display_labels=['Normal', 'Malicious']).plot(ax=ax1, cmap=plt.cm.Blues, values_format='d')
    ax1.set_title('ML Model')
    
    ConfusionMatrixDisplay(confusion_matrix=grl_cm, display_labels=['Normal', 'Malicious']).plot(ax=ax2, cmap=plt.cm.Blues, values_format='d')
    ax2.set_title('GRL Model')
    
    plt.suptitle('Confusion Matrix Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/combined_confusion_matrices.png")
    plt.close()


def plot_comparative_metrics(comparison, save_dir="plots"):
    """
    Plot comparative metrics between ML and GRL models.
    
    Args:
        comparison: Dict with comparison metrics
        save_dir: Directory to save the plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating comparative metric plots...")
    
    # Extract metrics for comparison
    metrics = ['Precision', 'Recall', 'Weighted_F1', 'Accuracy']
    ml_values = [comparison['ML'][m] for m in metrics]
    grl_values = [comparison['GRL'][m] for m in metrics]
    
    # Bar plot of metrics
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, ml_values, width, label='Informer Model', color='#3498db')
    bars2 = plt.bar(x + width/2, grl_values, width, label='GRL Model', color='#2ecc71')
    
    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Informer vs GRL Performance Comparison')
    plt.xticks(x, metrics)
    plt.ylim(0, max(max(ml_values), max(grl_values)) * 1.15)  # Add space for labels
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ml_vs_grl_metrics.png")
    plt.close()
    
    # Error type comparison (FP and FN)
    error_types = ['False Positives', 'False Negatives']
    ml_errors = [comparison['ML']['FP'], comparison['ML']['FN']]
    grl_errors = [comparison['GRL']['FP'], comparison['GRL']['FN']]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(error_types))
    
    bars1 = plt.bar(x - width/2, ml_errors, width, label='ML Model', color='#e74c3c')
    bars2 = plt.bar(x + width/2, grl_errors, width, label='GRL Model', color='#9b59b6')
    
    # Add value labels on top of bars
    add_labels(bars1)
    add_labels(bars2)
    
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.title('ML vs GRL Error Comparison')
    plt.xticks(x, error_types)
    plt.ylim(0, max(max(ml_errors), max(grl_errors)) * 1.15)  # Add space for labels
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ml_vs_grl_errors.png")
    plt.close()
    
    # Improvements bar chart
    improvements = comparison['GRL_vs_Rule_Based']
    labels = ['FP Reduction', 'FN Reduction', 'Weighted_F1_Improvement', 'Accuracy Improvement']
    values = [improvements['FP_Reduction'], improvements['FN_Reduction'], 
              improvements['Weighted_F1_Improvement'], improvements['Accuracy_Improvement']]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color=['#1abc9c', '#3498db', '#f39c12', '#9b59b6'])
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + 0.5 if height > 0 else height - 2.5,
                f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10)
    
    plt.xlabel('Metric')
    plt.ylabel('Improvement (%)')
    plt.title('GRL Improvements Over ML Model')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/grl_improvements.png")
    plt.close()
    
    # TP/TN comparison with pie charts
    plt.figure(figsize=(16, 8))
    
    # ML Model classification breakdown
    plt.subplot(1, 2, 1)
    ml_values = [comparison['ML']['TP'], comparison['ML']['TN'], 
                comparison['ML']['FP'], comparison['ML']['FN']]
    labels = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    plt.pie(ml_values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('ML Model Classification Breakdown')
    
    # GRL Model classification breakdown
    plt.subplot(1, 2, 2)
    grl_values = [comparison['GRL']['TP'], comparison['GRL']['TN'], 
                 comparison['GRL']['FP'], comparison['GRL']['FN']]
    plt.pie(grl_values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('GRL Model Classification Breakdown')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/classification_breakdown.png")
    plt.close()

def plot_action_distribution(metrics_path, save_dir="plots"):
    """
    Plot action distribution over training.
    
    Args:
        metrics_path: Path to the JSON file with training metrics
        save_dir: Directory to save the plots
    """
    # Check if file exists
    if not os.path.exists(metrics_path):
        print(f"Warning: {metrics_path} not found. Cannot plot action distribution.")
        return
    
    # Load the metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Check if we have action distribution data
    if "action_distribution" not in metrics or not metrics["action_distribution"] or all(sum(dist.values()) == 0 for dist in metrics["action_distribution"]):
        print("No action distribution data available. Creating placeholder plot instead.")
        plt.figure(figsize=(12, 6))
        plt.title('Action Distribution During Training (No Data Available)')
        plt.xlabel('Timesteps')
        plt.ylabel('Percentage (%)')
        plt.grid(True, alpha=0.3)
        plt.text(0.5, 0.5, 'No action distribution data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/action_distribution.png")
        plt.close()
        return
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating action distribution plots...")
    
    # Extract action distributions over time
    action_dist = metrics["action_distribution"]
    timesteps = metrics.get("timesteps", list(range(len(action_dist))))
    
    # Create arrays for plotting
    action0_counts = [dist.get("0", 0) for dist in action_dist]
    action1_counts = [dist.get("1", 0) for dist in action_dist]
    
    # Convert to percentages
    total_actions = [a0 + a1 for a0, a1 in zip(action0_counts, action1_counts)]
    action0_pct = [a0 / max(t, 1) * 100 for a0, t in zip(action0_counts, total_actions)]
    action1_pct = [a1 / max(t, 1) * 100 for a1, t in zip(action1_counts, total_actions)]
    
    # Plot action distribution over time
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, action0_pct, label='Action 0 (Prune)', color='#e74c3c')
    plt.plot(timesteps, action1_pct, label='Action 1 (Keep)', color='#2ecc71')
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    plt.title('Action Distribution During Training')
    plt.xlabel('Timesteps')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/action_distribution.png")
    plt.close()
    
    # Check if we have actions by label data
    if "actions_by_label" in metrics and metrics["actions_by_label"]:
        # Extract actions by label data
        actions_by_label = metrics["actions_by_label"]
        
        # Check if we have valid data for each label and action
        has_data = (
            "0" in actions_by_label and 
            "1" in actions_by_label and
            "0" in actions_by_label["0"] and 
            "1" in actions_by_label["0"] and
            "0" in actions_by_label["1"] and 
            "1" in actions_by_label["1"] and
            actions_by_label["0"]["0"] and  # List not empty
            actions_by_label["0"]["1"] and
            actions_by_label["1"]["0"] and
            actions_by_label["1"]["1"]
        )
        
        if has_data:
            # Plot actions taken for each label type
            plt.figure(figsize=(15, 10))
            
            # For normal edges (label 0)
            plt.subplot(2, 1, 1)
            normal_keep = actions_by_label["0"]["1"]
            normal_prune = actions_by_label["0"]["0"]
            
            # If lists have different lengths, pad with zeros
            max_len = max(len(normal_keep), len(normal_prune))
            normal_keep = normal_keep + [0] * (max_len - len(normal_keep))
            normal_prune = normal_prune + [0] * (max_len - len(normal_prune))
            
            plt.stackplot(range(len(normal_keep)), 
                        [normal_keep, normal_prune],
                        labels=['Keep (Correct)', 'Prune (Incorrect)'],
                        colors=['#2ecc71', '#e74c3c'])
            plt.title('Actions Taken for Normal Edges (Label 0)')
            plt.xlabel('Training Progress')
            plt.ylabel('Count')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # For malicious edges (label 1)
            plt.subplot(2, 1, 2)
            malicious_keep = actions_by_label["1"]["1"]
            malicious_prune = actions_by_label["1"]["0"]
            
            # If lists have different lengths, pad with zeros
            max_len = max(len(malicious_keep), len(malicious_prune))
            malicious_keep = malicious_keep + [0] * (max_len - len(malicious_keep))
            malicious_prune = malicious_prune + [0] * (max_len - len(malicious_prune))
            
            plt.stackplot(range(len(malicious_prune)), 
                        [malicious_prune, malicious_keep],
                        labels=['Prune (Correct)', 'Keep (Incorrect)'],
                        colors=['#2ecc71', '#e74c3c'])
            plt.title('Actions Taken for Malicious Edges (Label 1)')
            plt.xlabel('Training Progress')
            plt.ylabel('Count')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/actions_by_label.png")
            plt.close()
            
            # Plot correct action percentage for each label
            plt.figure(figsize=(12, 6))
            
            # Calculate percentage of correct actions for each label
            normal_correct_pct = []
            for i in range(len(normal_keep)):
                total = normal_keep[i] + normal_prune[i]
                if total > 0:
                    # For normal edges, "keep" (action 1) is correct
                    normal_correct_pct.append(normal_keep[i] / total * 100)
                else:
                    normal_correct_pct.append(0)
            
            malicious_correct_pct = []
            for i in range(len(malicious_prune)):
                total = malicious_keep[i] + malicious_prune[i]
                if total > 0:
                    # For malicious edges, "prune" (action 0) is correct
                    malicious_correct_pct.append(malicious_prune[i] / total * 100)
                else:
                    malicious_correct_pct.append(0)
            
            plt.plot(range(len(normal_correct_pct)), normal_correct_pct, 
                     label='Normal Edges', color='#3498db', linewidth=2)
            plt.plot(range(len(malicious_correct_pct)), malicious_correct_pct, 
                     label='Malicious Edges', color='#e74c3c', linewidth=2)
            plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            plt.title('Correct Action Percentage During Training')
            plt.xlabel('Training Progress')
            plt.ylabel('Correct Action Percentage (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/correct_action_percentages.png")
            plt.close()
        else:
            print("No valid actions by label data available. Skipping actions_by_label plots.")
    else:
        print("No actions by label data available. Skipping actions_by_label plots.")



if __name__ == "__main__":
    # File paths
    metrics_path = "logs/training_metrics.json"
    
    # For demonstration, let's create some sample comparison data
    sample_comparison = {
        'ML': {
            'Precision': 0.75,
            'Recall': 0.80,
            'F1': 0.77,
            'Accuracy': 0.82,
            'FP': 50,
            'FN': 40,
            'TP': 150,
            'TN': 200,
            'Confusion Matrix': np.array([[200, 50], [40, 150]])
        },
        'GRL': {
            'Precision': 0.85,
            'Recall': 0.88,
            'F1': 0.86,
            'Accuracy': 0.90,
            'FP': 30,
            'FN': 20,
            'TP': 170,
            'TN': 220,
            'Confusion Matrix': np.array([[220, 30], [20, 170]])
        },
        'Improvements': {
            'FP_Reduction': 40.0,
            'FN_Reduction': 50.0,
            'F1_Improvement': 11.7,
            'Accuracy_Improvement': 9.8
        }
    }
    
    # Plot training metrics if the file exists
    if os.path.exists(metrics_path):
        print("Plotting training metrics...")
        plot_training_metrics(metrics_path)
        plot_improved_training_rewards(metrics_path)
    else:
        print(f"Training metrics file {metrics_path} not found. Skipping training plots.")
    
    # Plot comparison metrics
    print("Plotting comparison metrics...")
    plot_confusion_matrices(sample_comparison)
    plot_comparative_metrics(sample_comparison)
    
    print("Visualization complete! Check the 'plots' directory for output images.")

def plot_rule_based_comparison(comparison, save_dir="plots"):
    """
    Plot comparison between ML, GRL, and Rule-Based approaches.
    
    Args:
        comparison: Dict with comparison metrics including Rule_Based
        save_dir: Directory to save the plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    if 'Rule_Based' not in comparison:
        print("No Rule-Based metrics found in comparison data.")
        return
    
    # Extract metrics for comparison
    metrics = ['Precision', 'Recall', 'Weighted_F1', 'Accuracy']
    ml_values = [comparison['ML'][m] for m in metrics]
    grl_values = [comparison['GRL'][m] for m in metrics]
    rule_values = [comparison['Rule_Based'][m.lower()] for m in metrics]
    
    # Bar plot of metrics
    plt.figure(figsize=(12, 7))
    x = np.arange(len(metrics))
    width = 0.25
    
    bars1 = plt.bar(x - width, ml_values, width, label='ML Model', color='#3498db')
    bars2 = plt.bar(x, rule_values, width, label='Rule-Based', color='#e74c3c')
    bars3 = plt.bar(x + width, grl_values, width, label='GRL Model', color='#2ecc71')
    
    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison: ML vs Rule-Based vs GRL')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/three_model_comparison.png")
    plt.close()
    
    # Error type comparison (FP and FN)
    error_types = ['False Positives', 'False Negatives']
    ml_errors = [comparison['ML']['FP'], comparison['ML']['FN']]
    rule_errors = [comparison['Rule_Based']['fp'], comparison['Rule_Based']['fn']]
    grl_errors = [comparison['GRL']['FP'], comparison['GRL']['FN']]
    
    plt.figure(figsize=(12, 7))
    x = np.arange(len(error_types))
    
    bars1 = plt.bar(x - width, ml_errors, width, label='ML Model', color='#3498db')
    bars2 = plt.bar(x, rule_errors, width, label='Rule-Based', color='#e74c3c')
    bars3 = plt.bar(x + width, grl_errors, width, label='GRL Model', color='#2ecc71')
    
    # Add value labels on top of bars
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.title('Error Comparison: ML vs Rule-Based vs GRL')
    plt.xticks(x, error_types)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/three_model_errors.png")
    plt.close()
    
    # GRL vs Rule-Based improvements chart
    if 'GRL_vs_Rule_Based' in comparison:
        improvements = comparison['GRL_vs_Rule_Based']
        labels = ['FP Reduction', 'FN Reduction', 'Weighted_F1_Improvement', 'Accuracy Improvement']
        values = [improvements['FP_Reduction'], improvements['FN_Reduction'], 
                improvements['Weighted_F1_Improvement'], improvements['Accuracy_Improvement']]
        
        plt.figure(figsize=(12, 7))
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
        bars = plt.bar(labels, values, color=colors)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 1 if height >= 0 else height - 4,
                    f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10)
        
        plt.ylabel('Improvement (%)')
        plt.title('GRL Improvements Over Rule-Based Approach')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/grl_vs_rule_based.png")
        plt.close()

def plot_reward_breakdown(metrics, save_dir="plots"):
    if "reward_breakdown" not in metrics:
        print("No reward breakdown data found.")
        return

    rb = metrics["reward_breakdown"]
    timesteps = metrics.get("timesteps", range(len(rb["tp"])))

    plt.figure(figsize=(14, 6))
    plt.plot(timesteps, rb["tp"], label="TP (Correct Prune)", color="#2ecc71")
    plt.plot(timesteps, rb["tn"], label="TN (Correct Keep)", color="#3498db")
    plt.plot(timesteps, rb["fp"], label="FP (Wrong Prune)", color="#e74c3c")
    plt.plot(timesteps, rb["fn"], label="FN (Missed Malicious)", color="#f39c12")
    
    plt.title("Reward Breakdown During Training")
    plt.xlabel("Timesteps")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reward_breakdown.png")
    plt.close()

def plot_smooth_training_rewards(metrics, save_dir="plots"):
    if "rewards" not in metrics or not metrics["rewards"]:
        print("No reward data found.")
        return

    rewards = metrics["rewards"]
    timesteps = list(range(len(rewards)))

    # Plot raw reward + smoothed moving average
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, rewards, label="Raw Reward", color="lightblue", alpha=0.4)

    # Moving average (e.g., 50-episode window)
    window = 50
    smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
    plt.plot(timesteps, smoothed, label=f"Smoothed Reward (window={window})", color="blue", linewidth=2)

    plt.title("Smoothed Episode Rewards During Training")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/smoothed_training_rewards.png")
    plt.close()

