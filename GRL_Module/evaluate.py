import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env import VANETGraphEnv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import os
from tqdm.auto import tqdm
import json
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

# Define a simple wrapper function that pads observations
def pad_observation(obs, target_size=460):
    """
    Adjust observation shape to match target size:
    - If obs is smaller than target_size, pad with zeros
    - If obs is larger than target_size, truncate
    """
    if isinstance(obs, np.ndarray):
        # Single observation
        if len(obs.shape) == 1:
            if obs.shape[0] < target_size:
                # Pad with zeros
                padded = np.zeros(target_size, dtype=np.float32)
                padded[:obs.shape[0]] = obs
                return padded
            elif obs.shape[0] > target_size:
                # Truncate
                return obs[:target_size]
        # Batched observation
        elif len(obs.shape) == 2:
            if obs.shape[1] < target_size:
                # Pad with zeros
                padded = np.zeros((obs.shape[0], target_size), dtype=np.float32)
                padded[:, :obs.shape[1]] = obs
                return padded
            elif obs.shape[1] > target_size:
                # Truncate
                return obs[:, :target_size]
    return obs

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


def evaluate_model(model, env_path, num_episodes=10, threshold=0.5):
    """
    Evaluate the trained model and collect performance metrics.
    
    Args:
        model: Trained PPO model
        env_path: Path to the data file
        num_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create environment
    env = VANETGraphEnv(env_path)

    # Tell VecNormalize not to update running statistics during evaluation
    env.training = False
    
    # If you have model's VecNormalize statistics, you can sync them:
    if hasattr(model, 'get_vec_normalize_env') and model.get_vec_normalize_env() is not None:
        orig_vec_normalize = model.get_vec_normalize_env()
        env.obs_rms = orig_vec_normalize.obs_rms
        env.ret_rms = orig_vec_normalize.ret_rms
    
    # Track metrics
    total_rewards = []
    all_true_labels = []
    all_actions = []
    all_ml_predictions = []
    all_raw_actions = [] # Store pre-masked actions for threshold analysis

    # Additional tracking for detailed analysis
    all_confidence_scores = []  # Store confidence scores for edges
    all_communication_history = []  # Store communication history
    
    # Setup progress bar
    pbar = tqdm(total=num_episodes, desc="Evaluating Episodes")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        # Pad observation to match expected shape
        padded_obs = pad_observation(obs, target_size=460)
        batched_obs = np.expand_dims(padded_obs, axis=0) if len(padded_obs.shape) == 1 else padded_obs

        
        while not done:
            # Get action from model
            raw_actionn, _states = model.predict(batched_obs, deterministic=True)

            # Remove batch dimension for environment step
            raw_action = raw_actionn[0]

            masked_action = env._apply_action_masking(raw_action)
            
            # Track predictions and true labels
            all_true_labels.extend(env.current_labels[:len(masked_action)])
            all_actions.extend([int(a) for a in masked_action[:len(env.current_labels)]])
            all_ml_predictions.extend(env.current_predictions[:len(masked_action)])
            all_raw_actions.extend([float(a) for a in raw_action[:len(env.current_labels)]])

            # Track additional metrics if available
            if hasattr(env, 'prediction_confidence'):
                for sender, receiver in env.current_edges[:len(masked_action)]:
                    pair_key = (sender, receiver)
                    confidence = env.prediction_confidence.get(pair_key, 0.5)
                    history = env.communication_history.get(pair_key, 0.0)
                    all_confidence_scores.append(confidence)
                    all_communication_history.append(history)
            
            # Take step
            obs, reward, done, _, info = env.step(masked_action)
            episode_reward += reward
        
            # Pad observation to match expected shape
            padded_obs = pad_observation(obs, target_size=460)
            batched_obs = np.expand_dims(padded_obs, axis=0) if len(padded_obs.shape) == 1 else padded_obs
        
        # Record episode reward
        total_rewards.append(float(episode_reward))  # Convert to standard float for JSON serialization
        
        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    # Convert to numpy arrays for easier analysis
    true_labels = np.array(all_true_labels)
    grl_actions = np.array(all_actions)
    ml_predictions = np.array(all_ml_predictions)
    raw_actions = np.array(all_raw_actions)
    
    # For GRL, action=0 means "prune" (predict malicious), action=1 means "keep" (predict normal)
    # Need to invert since our dataset uses 1=malicious, 0=normal
    grl_predictions = 1 - grl_actions

    correct_actions = track_correct_actions(true_labels, ml_predictions, grl_actions)

    # Perform threshold analysis on raw actions
    thresholds = np.linspace(0, 1, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    threshold_metrics = analyze_thresholds(true_labels, raw_actions, thresholds)
    
    # Calculate metrics at different confidence levels
    confidence_metrics = None
    if all_confidence_scores:
        confidence_scores = np.array(all_confidence_scores)
        confidence_metrics = analyze_confidence_levels(true_labels, grl_predictions, confidence_scores)
    
    # Calculate metrics with additional context
    context_metrics = None
    if all_communication_history:
        history_values = np.array(all_communication_history)
        context_metrics = analyze_communication_history(true_labels, grl_predictions, history_values)
    
    # Calculate metrics
    evaluation_metrics = {
        'avg_reward': float(np.mean(total_rewards)),
        'true_labels': true_labels.tolist(),
        'grl_predictions': grl_predictions.tolist(),
        'ml_predictions': ml_predictions.tolist(),
        'raw_actions': raw_actions.tolist(),
        'correct_actions': correct_actions,
        'threshold_metrics': threshold_metrics,

        'weighted_f1_grl': calculate_weighted_f1_score(true_labels, grl_predictions),
        'weighted_f1_ml': calculate_weighted_f1_score(true_labels, ml_predictions),
    }

    
        # Add confidence metrics if available
    if confidence_metrics:
        evaluation_metrics['confidence_metrics'] = confidence_metrics
        
    # Add context metrics if available
    if context_metrics:
        evaluation_metrics['context_metrics'] = context_metrics
    
    return evaluation_metrics

def analyze_thresholds(true_labels, raw_actions, thresholds):
    """
    Analyze model performance at different decision thresholds.
    
    Args:
        true_labels: True labels (0=normal, 1=malicious)
        raw_actions: Raw action values before thresholding (0-1 range)
        thresholds: List of thresholds to evaluate
        
    Returns:
        Dictionary of metrics at each threshold
    """
    results = {}
    
    # Invert raw actions because action 0 (prune) means prediction 1 (malicious)
    raw_predictions = 1 - raw_actions
    
    # Calculate ROC and PR curves
    fpr, tpr, roc_thresholds = roc_curve(true_labels, raw_predictions)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, pr_thresholds = precision_recall_curve(true_labels, raw_predictions)
    pr_auc = auc(recall, precision)
    
    # Store curve metrics
    results['roc_curve'] = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': roc_thresholds.tolist(),
        'auc': float(roc_auc)
    }
    
    results['pr_curve'] = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else [],
        'auc': float(pr_auc)
    }
    
    # Evaluate each threshold
    results['thresholds'] = {}
    for threshold in thresholds:
        # Apply threshold to get binary predictions
        binary_predictions = (raw_predictions >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(true_labels, binary_predictions).ravel()
        
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        accuracy_val = (tp + tn) / (tp + tn + fp + fn)
        
        results['thresholds'][str(threshold)] = {
            'precision': float(precision_val),
            'recall': float(recall_val),
            'f1': float(f1_val),
            'accuracy': float(accuracy_val),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    # Find optimal threshold for F1 score
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold, metrics in results['thresholds'].items():
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = float(threshold)
    
    results['optimal_threshold'] = {
        'f1': float(best_f1),
        'threshold': float(best_threshold)
    }
    
    return results


def analyze_confidence_levels(true_labels, predictions, confidence_scores):
    """
    Analyze model performance at different confidence levels.
    
    Args:
        true_labels: True labels (0=normal, 1=malicious)
        predictions: Model predictions (0=normal, 1=malicious)
        confidence_scores: Confidence scores for each prediction (0-1 range)
        
    Returns:
        Dictionary of metrics at different confidence levels
    """
    # Define confidence bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_names = ['very_low', 'low', 'moderate', 'high', 'very_high']
    
    results = {}
    
    # Analyze each confidence bin
    for i in range(len(bins) - 1):
        bin_name = bin_names[i]
        lower = bins[i]
        upper = bins[i+1]
        
        # Filter by confidence level
        mask = (confidence_scores >= lower) & (confidence_scores < upper)
        if not np.any(mask):
            # Skip if no samples in this bin
            results[bin_name] = {'count': 0}
            continue
            
        bin_labels = true_labels[mask]
        bin_preds = predictions[mask]
        
        # Calculate metrics for this bin
        try:
            tn, fp, fn, tp = confusion_matrix(bin_labels, bin_preds).ravel()
            
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
            accuracy_val = (tp + tn) / (tp + tn + fp + fn)
            
            results[bin_name] = {
                'count': int(np.sum(mask)),
                'precision': float(precision_val),
                'recall': float(recall_val),
                'f1': float(f1_val),
                'accuracy': float(accuracy_val),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
        except Exception as e:
            # Handle case where confusion matrix fails (e.g., only one class present)
            results[bin_name] = {
                'count': int(np.sum(mask)),
                'error': str(e)
            }
    
    return results


def analyze_communication_history(true_labels, predictions, history_values):
    """
    Analyze model performance based on communication history.
    
    Args:
        true_labels: True labels (0=normal, 1=malicious)
        predictions: Model predictions (0=normal, 1=malicious)
        history_values: Communication history values for each edge
        
    Returns:
        Dictionary of metrics based on communication history
    """
    # Define history bins
    bins = [-1.0, -0.5, 0.0, 0.5, 1.0]
    bin_names = ['negative_high', 'negative_low', 'neutral', 'positive']
    
    results = {}
    
    # Analyze each history bin
    for i in range(len(bins) - 1):
        bin_name = bin_names[i]
        lower = bins[i]
        upper = bins[i+1]
        
        # Filter by history level
        mask = (history_values >= lower) & (history_values < upper)
        if not np.any(mask):
            # Skip if no samples in this bin
            results[bin_name] = {'count': 0}
            continue
            
        bin_labels = true_labels[mask]
        bin_preds = predictions[mask]
        
        # Calculate metrics for this bin
        try:
            tn, fp, fn, tp = confusion_matrix(bin_labels, bin_preds).ravel()
            
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
            accuracy_val = (tp + tn) / (tp + tn + fp + fn)
            
            results[bin_name] = {
                'count': int(np.sum(mask)),
                'precision': float(precision_val),
                'recall': float(recall_val),
                'f1': float(f1_val),
                'accuracy': float(accuracy_val),
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
        except Exception as e:
            # Handle case where confusion matrix fails
            results[bin_name] = {
                'count': int(np.sum(mask)),
                'error': str(e)
            }
    
    return results


def compare_ml_vs_grl(metrics):
    """
    Compare the ML model alone vs ML+GRL performance and rule-based approach
    
    Args:
        metrics: Dictionary with evaluation metrics
    
    Returns:
        Dictionary with comparison results
    """
    # Convert lists back to numpy arrays if needed
    true_labels = np.array(metrics['true_labels']) if isinstance(metrics['true_labels'], list) else metrics['true_labels']
    ml_predictions = np.array(metrics['ml_predictions']) if isinstance(metrics['ml_predictions'], list) else metrics['ml_predictions']
    grl_predictions = np.array(metrics['grl_predictions']) if isinstance(metrics['grl_predictions'], list) else metrics['grl_predictions']
            

    # Calculate confusion matrices
    grl_cm = confusion_matrix(true_labels, grl_predictions)
    ml_cm = confusion_matrix(true_labels, ml_predictions)
    
    # Extract values from confusion matrices
    ml_tn, ml_fp, ml_fn, ml_tp = ml_cm.ravel()
    grl_tn, grl_fp, grl_fn, grl_tp = grl_cm.ravel()

    # Convert NumPy integers to standard Python integers
    ml_tn, ml_fp, ml_fn, ml_tp = int(ml_tn), int(ml_fp), int(ml_fn), int(ml_tp)
    grl_tn, grl_fp, grl_fn, grl_tp = int(grl_tn), int(grl_fp), int(grl_fn), int(grl_tp)
    
    # Get rule-based metrics
    rule_based_metrics = evaluate_rule_based_approach(true_labels, ml_predictions)
    rule_tp = rule_based_metrics['tp']
    rule_fp = rule_based_metrics['fp']
    rule_tn = rule_based_metrics['tn']
    rule_fn = rule_based_metrics['fn']
    
    # Calculate metrics for ML
    ml_precision = ml_tp / (ml_tp + ml_fp) if (ml_tp + ml_fp) > 0 else 0
    ml_recall = ml_tp / (ml_tp + ml_fn) if (ml_tp + ml_fn) > 0 else 0
    ml_f1 = 2 * (ml_precision * ml_recall) / (ml_precision + ml_recall) if (ml_precision + ml_recall) > 0 else 0
    ml_accuracy = (ml_tp + ml_tn) / (ml_tp + ml_tn + ml_fp + ml_fn)

    # Calculate WEIGHTED metrics for ML
    ml_precision_W = calculate_weighted_precision(true_labels, ml_predictions)
    ml_recall_W = calculate_weighted_recall(true_labels, ml_predictions)
    ml_f1_W = calculate_weighted_f1_score(true_labels, ml_predictions)
    
    # Calculate metrics for GRL
    grl_precision = grl_tp / (grl_tp + grl_fp) if (grl_tp + grl_fp) > 0 else 0
    grl_recall = grl_tp / (grl_tp + grl_fn) if (grl_tp + grl_fn) > 0 else 0
    grl_f1 = 2 * (grl_precision * grl_recall) / (grl_precision + grl_recall) if (grl_precision + grl_recall) > 0 else 0
    grl_accuracy = (grl_tp + grl_tn) / (grl_tp + grl_tn + grl_fp + grl_fn)

    # Calculate WEIGHTED metrics for GRL
    grl_precision_W = calculate_weighted_precision(true_labels, grl_predictions)
    grl_recall_W = calculate_weighted_recall(true_labels, grl_predictions)
    grl_f1_W = calculate_weighted_f1_score(true_labels, grl_predictions)

    
    # Compile results with standard Python types (for JSON serialization)
    comparison = {
        'ML': {
            'Precision': float(ml_precision),
            'Precision_W': float(ml_precision_W),
            'Recall': float(ml_recall),
            'Recall_W': float(ml_recall_W),
            'F1': float(ml_f1),
            'F1_W': float(ml_f1_W),
            'Accuracy': float(ml_accuracy),
            'FP': ml_fp,
            'FN': ml_fn,
            'TP': ml_tp,
            'TN': ml_tn,
            'Confusion Matrix': ml_cm.tolist()
        },
        'GRL': {
            'Precision': float(grl_precision),
            'Precision_W': float(grl_precision_W),
            'Recall': float(grl_recall),
            'Recall_W': float(grl_recall_W),
            'F1': float(grl_f1),
            'F1_W': float(grl_f1_W),
            'Accuracy': float(grl_accuracy),
            'FP': grl_fp,
            'FN': grl_fn,
            'TP': grl_tp,
            'TN': grl_tn,
            'Confusion Matrix': grl_cm.tolist()
        }
    }
    
    # Calculate GRL improvements over rule-based
    fp_improvement = (rule_fp - grl_fp) / rule_fp * 100 if rule_fp > 0 else 0
    fn_improvement = (rule_fn - grl_fn) / rule_fn * 100 if rule_fn > 0 else 0
    f1_improvement = (grl_f1 - rule_based_metrics['f1']) / rule_based_metrics['f1'] * 100 if rule_based_metrics['f1'] > 0 else 0
    f1_improvement_W = (grl_f1_W - rule_based_metrics['f1']) / rule_based_metrics['f1'] * 100 if rule_based_metrics['f1'] > 0 else 0
    accuracy_improvement = (grl_accuracy - rule_based_metrics['accuracy']) / rule_based_metrics['accuracy'] * 100 if rule_based_metrics['accuracy'] > 0 else 0

    
    comparison['GRL_vs_Rule_Based'] = {
        'FP_Reduction': float(fp_improvement),
        'FN_Reduction': float(fn_improvement),
        'F1_Improvement': float(f1_improvement),
        'F1_Improvement_W': float(f1_improvement_W),
        'Accuracy_Improvement': float(accuracy_improvement)
    }

    # Add weighted F1 to the comparison results
    comparison['ML']['Weighted_F1'] = metrics['weighted_f1_ml']
    comparison['GRL']['Weighted_F1'] = metrics['weighted_f1_grl']

    # Calculate weighted F1 for rule-based approach (same as ML)
    comparison['Rule_Based']['weighted_f1'] = metrics['weighted_f1_ml']
    
    # Calculate improvements for weighted F1
    weighted_f1_improvement = ((metrics['weighted_f1_grl'] - metrics['weighted_f1_ml']) 
                              / metrics['weighted_f1_ml'] * 100) if metrics['weighted_f1_ml'] > 0 else 0
    
    
    comparison['GRL_vs_Rule_Based']['Weighted_F1_Improvement'] = float(weighted_f1_improvement)
    
    return comparison


if __name__ == "__main__":
    # File paths
    data_path = "vanet_data.csv"
    model_path = "models/vanet_grl_model"
    
    # Create output directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(model, data_path)
    
    # Compare ML vs GRL
    print("Comparing ML vs GRL...")
    comparison = compare_ml_vs_grl(metrics)
    
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
    for k, v in comparison['Improvements'].items():
        print(f"  {k}: {v:.2f}%")
    
    print("\nEvaluation complete!")

def evaluate_rule_based_approach(true_labels, ml_predictions):
    """
    Evaluate a rule-based approach that directly uses ML predictions.
    Rule: If prediction is 1 (malicious), prune (action=0). If prediction is 0 (normal), keep (action=1).
    
    Args:
        true_labels: True labels (0=normal, 1=malicious)
        ml_predictions: ML predictions (0=normal, 1=malicious)
    
    Returns:
        Dictionary with performance metrics
    """
    # Rule-based approach: action matches ML prediction
    # If ML predicts 1 (malicious), then prune (action=0)
    # If ML predicts 0 (normal), then keep (action=1)
    rule_based_actions = 1 - ml_predictions  # Invert because action 0=prune, 1=keep
    rule_based_predictions = ml_predictions  # For rule-based, predictions = ML predictions
    
    # Calculate confusion matrix
    rule_cm = confusion_matrix(true_labels, rule_based_predictions)
    rule_tn, rule_fp, rule_fn, rule_tp = rule_cm.ravel()

    rule_tn, rule_fp, rule_fn, rule_tp = int(rule_tn), int(rule_fp), int(rule_fn), int(rule_tp)

    # Calculate metrics
    rule_precision = rule_tp / (rule_tp + rule_fp) if (rule_tp + rule_fp) > 0 else 0
    rule_recall = rule_tp / (rule_tp + rule_fn) if (rule_tp + rule_fn) > 0 else 0
    rule_f1 = 2 * rule_precision * rule_recall / (rule_precision + rule_recall) if (rule_precision + rule_recall) > 0 else 0
    rule_accuracy = (rule_tp + rule_tn) / (rule_tp + rule_tn + rule_fp + rule_fn)

    rule_precision = float(rule_precision)
    rule_recall = float(rule_recall)
    rule_f1 = float(rule_f1)
    rule_accuracy = float(rule_accuracy)
    
    return {
        'precision': float(rule_precision),
        'recall': float(rule_recall),
        'f1': float(rule_f1),
        'accuracy': float(rule_accuracy),
        'tp': int(rule_tp),
        'fp': int(rule_fp),
        'tn': int(rule_tn),
        'fn': int(rule_fn),
        'confusion_matrix': rule_cm.tolist()
    }

def track_correct_actions(true_labels, predictions, actions):
    """
    Track correct action percentages by edge type during evaluation.
    
    Args:
        true_labels: Array of true labels (0=normal, 1=malicious)
        predictions: Array of model predictions
        actions: Array of actions taken (0=prune, 1=keep)
    
    Returns:
        Dictionary with correct action percentages
    """
    normal_indices = (true_labels == 0)
    malicious_indices = (true_labels == 1)
    
    # For normal edges (label=0), correct action is keep (1)
    normal_total = np.sum(normal_indices)
    normal_correct = np.sum(actions[normal_indices] == 1)
    normal_correct_pct = (normal_correct / normal_total * 100) if normal_total > 0 else 0
    
    # For malicious edges (label=1), correct action is prune (0)
    malicious_total = np.sum(malicious_indices)
    malicious_correct = np.sum(actions[malicious_indices] == 0)
    malicious_correct_pct = (malicious_correct / malicious_total * 100) if malicious_total > 0 else 0
    
    overall_correct = (normal_correct + malicious_correct) / (normal_total + malicious_total) * 100
    
    return {
        'normal_correct_pct': float(normal_correct_pct),
        'malicious_correct_pct': float(malicious_correct_pct),
        'overall_correct_pct': float(overall_correct),
        'normal_correct': int(normal_correct),
        'normal_total': int(normal_total),
        'malicious_correct': int(malicious_correct),
        'malicious_total': int(malicious_total)
    }
    

def calculate_weighted_f1_score(true_labels, predictions):
    """
    Calculate the weighted F1 score similar to sklearn's f1_score(y_true, y_pred, average='weighted')
    
    Args:
        true_labels: True labels (0=normal, 1=malicious)
        predictions: Model predictions (0=normal, 1=malicious)
        
    Returns:
        Float: The weighted F1 score
    """
    # Convert to numpy arrays if needed
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    # Get unique classes
    classes = np.unique(true_labels)
    
    # Calculate class weights based on number of samples in each class
    class_weights = {}
    total_samples = len(true_labels)
    
    for c in classes:
        class_weights[int(c)] = np.sum(true_labels == c) / total_samples
    
    # Calculate per-class metrics
    class_metrics = {}
    
    for c in classes:
        c = int(c)
        # True positives: predictions == c and true_labels == c
        tp = np.sum((predictions == c) & (true_labels == c))
        
        # False positives: predictions == c and true_labels != c
        fp = np.sum((predictions == c) & (true_labels != c))
        
        # False negatives: predictions != c and true_labels == c
        fn = np.sum((predictions != c) & (true_labels == c))
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[c] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'weight': float(class_weights[c]),
            'samples': int(np.sum(true_labels == c))
        }
    
    # Calculate weighted F1 score
    weighted_f1 = 0
    for c in classes:
        c = int(c)
        weighted_f1 += class_metrics[c]['f1'] * class_metrics[c]['weight']
    
    return float(weighted_f1)


def calculate_weighted_precision(true_labels, predictions):
    """
    Calculate the weighted precision score similar to sklearn's precision_score(y_true, y_pred, average='weighted')
    
    Args:
        true_labels: True labels (0=normal, 1=malicious)
        predictions: Model predictions (0=normal, 1=malicious)
        
    Returns:
        Float: The weighted precision score
    """
    # Convert to numpy arrays if needed
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    # Get unique classes
    classes = np.unique(true_labels)
    
    # Calculate class weights based on number of samples in each class
    class_weights = {}
    total_samples = len(true_labels)
    
    for c in classes:
        class_weights[int(c)] = np.sum(true_labels == c) / total_samples
    
    # Calculate per-class precision
    weighted_precision = 0
    
    for c in classes:
        c = int(c)
        # True positives: predictions == c and true_labels == c
        tp = np.sum((predictions == c) & (true_labels == c))
        
        # False positives: predictions == c and true_labels != c
        fp = np.sum((predictions == c) & (true_labels != c))
        
        # Calculate precision for this class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Add weighted contribution
        weighted_precision += precision * class_weights[c]
    
    return float(weighted_precision)


def calculate_weighted_recall(true_labels, predictions):
    """
    Calculate the weighted recall score similar to sklearn's recall_score(y_true, y_pred, average='weighted')
    
    Args:
        true_labels: True labels (0=normal, 1=malicious)
        predictions: Model predictions (0=normal, 1=malicious)
        
    Returns:
        Float: The weighted recall score
    """
    # Convert to numpy arrays if needed
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    # Get unique classes
    classes = np.unique(true_labels)
    
    # Calculate class weights based on number of samples in each class
    class_weights = {}
    total_samples = len(true_labels)
    
    for c in classes:
        class_weights[int(c)] = np.sum(true_labels == c) / total_samples
    
    # Calculate per-class recall
    weighted_recall = 0
    
    for c in classes:
        c = int(c)
        # True positives: predictions == c and true_labels == c
        tp = np.sum((predictions == c) & (true_labels == c))
        
        # False negatives: predictions != c and true_labels == c
        fn = np.sum((predictions != c) & (true_labels == c))
        
        # Calculate recall for this class
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Add weighted contribution
        weighted_recall += recall * class_weights[c]
    
    return float(weighted_recall)