# Multi-Agent Deep Reinforcement Learning for Energy-Efficient 5G Networks

This repository contains an advanced industry-grade simulation framework leveraging **Multi-Agent Deep Reinforcement Learning (MARL)** to autonomously organize and optimize 5G Heterogeneous Networks (HetNets).

## Architecture

The project replaces traditional rule-based SONs (Self-Organizing Networks) with an intelligent, decentralized multi-agent system built on **PyTorch**, featuring:
- **Continuous State Observation**: Base stations monitor real-time stochastic user mobility, traffic load fluctuations, SINR (Interference), and spectral throughput.
- **Deep Q-Networks (DQN)**: Autonomous agents structured as multi-layer perceptrons utilizing Experience Replay buffers and Target Networks for stable convergence, replacing naive tabular Q-learning.
- **Optimization Strategy**: Autonomous transition between Macrocell sleeping states (`Sleep`, `Low Power Idle`, `Active TX`) mitigating dynamic energy waste while severely penalizing transmission loss to guarantee Quality of Service (QoS).

## Repository Structure

```text
├── config/
│   └── settings.json         # Hyperparameter configurations (epsilon decay, gamma, batch sizes)
├── src/
│   ├── agent/
│   │   ├── dqn_agent.py      # Core DQN logic, epsilon-greedy exploration, Bellman updates
│   │   ├── models.py         # PyTorch Neural Network architectures
│   │   └── replay_buffer.py  # Experience memory buffer for stochastic sampling
│   ├── env/
│   │   └── network_env.py    # Emulates 5G kinematics (Users, Load, Bandwidth, Pathloss)
│   └── utils/
│       └── plotter.py        # Advanced Seaborn trajectory generation
├── main.py                   # Entrypoint with CLI arguments 
├── requirements.txt          # Python dependencies
└── training.log              # Persisted output logs
```

## Running the Simulation

Ensure your environment is configured up to specifications before initiating the deep learning tasks.
```bash
pip install -r requirements.txt
python main.py --config config/settings.json
```
