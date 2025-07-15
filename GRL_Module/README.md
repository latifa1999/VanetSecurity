# Graph Reinforcement Learning (GRL) Module
This module implements a Graph Reinforcement Learning approach for intelligent link pruning in VANETs. It combines Proximal Policy Optimization (PPO) with Graph Neural Networks (GNN) to make security-aware decisions about maintaining or pruning communication links based on network topology and anomaly detection results.

## Overview
The GRL module provides:

- Graph-based VANET Environment: Models vehicular networks as dynamic graphs
- PPO Training: Policy gradient method for decision making
- GNN Integration: Graph neural networks for topology-aware features
- Link Pruning Actions: Binary decisions to maintain or prune communication links
- Performance Evaluation: Comprehensive metrics for security and connectivity

## Directory Structure
<img width="626" height="225" alt="image" src="https://github.com/user-attachments/assets/8689503a-20b2-4786-a8ab-efc8e1479f9a" />

## Setup Environment
```
# Create virtual environment
python -m venv grl_env
source grl_env/bin/activate  # On Windows: grl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# training
python main.py \
  --learning_rate 0.0003 \
  --batch_size 64 \
  --n_epochs 10

# Training with Hyperparameter Optimization
python hyperopt.py \
  --n_trials 100

# Evaluate Trained Model
python main.py \
  --evaluate

# visualize
python main.py \
  --visualize
```
