# Securing VANETs Using a Digital Twin with Anomaly Detection and Graph Reinforcement Learning
A real-time, simulation-driven VANET security framework that combines digital twin technology, time-series anomaly detection, and graph reinforcement learning to detect and mitigate malicious attacks in vehicular networks.

## Overview
This project implements a comprehensive security solution for Vehicular Ad-hoc Networks (VANETs) using a multi-layered approach:

- Digital Twin Environment: Real-world traffic simulation using SUMO + OMNeT++ + Veins
- Anomaly Detection: Time-series models (Informer) for V2V message classification
- Graph Reinforcement Learning: PPO + GNN agent for intelligent link pruning
- Real-time Integration: Python-C++ socket communication for closed-loop execution

The system operates in real-time: the simulation generates V2V communication data, which is processed by the anomaly detection model to classify messages, and then the GRL agent makes pruning decisions to block malicious communication links.

## Architecture
<img width="840" height="712" alt="image" src="https://github.com/user-attachments/assets/5c2d745a-f31d-4209-b219-c61df6f24f31" />

## Repository Structure
1. Time_Series_Anomaly_Detection/: 
Contains implementation of state-of-the-art time-series models for V2V message anomaly detection:

- Multiple Model Support: Includes various time-series models for classification
- Data Processing: Custom data providers and preprocessing utilities
- Training Scripts: Complete training pipeline with hyperparameter tuning
- Evaluation Tools: Performance metrics and visualization scripts

2. GRL_Module/: 
Graph Reinforcement Learning implementation combining Proximal Policy Optimization (PPO) with Graph Neural Networks:

- Environment Setup: VANET graph environment simulation
- GNN Integration: Graph neural network for topology awareness
- PPO Training: Policy gradient method for link pruning decisions
- Evaluation Tools: Performance assessment and visualization

3. Simulation/Vanet/: 
OMNeT++ and SUMO configuration files for realistic vehicular network simulation:

- Multiple Road Scenarios: Three different traffic environments
- V2V Communication
- Attack Simulation: Configurable malicious behaviors
- Real-time Data Export: V2V message logging for analysis

4. Python_C++/: 
Python-C++ communication bridge enabling real-time integration:

- Socket Communication: TCP/IP bridge between simulation and Python
- Model Integration: Loads both AD and GRL trained models
- Real-time Processing: Low-latency message classification and decision making
- Action Execution: Commands sent back to simulation for link pruning

