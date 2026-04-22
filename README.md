# 🤖 Multi-Agent Deep Reinforcement Learning for Energy-Efficient 5G Networks

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-OpenAI-black)](https://gymnasium.farama.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Institution](https://img.shields.io/badge/IIIT-Vadodara-orange)](https://iiitvadodara.ac.in)

> **Course:** AI in 5G Networks | Indian Institute of Information Technology Vadodara  
> **Authors:** Rishikesh Gopal · Deep Bartaria · Shreya Tiwari · Abhi Trivedi

---

## 📌 Overview

This repository contains an industry-grade simulation framework leveraging **Multi-Agent Deep Reinforcement Learning (MARL)** to autonomously organize and optimize 5G Heterogeneous Networks (HetNets).

The project replaces traditional rule-based **Self-Organizing Networks (SON)** with an intelligent, decentralized multi-agent system where each Base Station (BS) acts as an independent DQN agent — learning to transition between hardware power states to minimize energy waste while guaranteeing Quality of Service (QoS).

### 🏆 Key Results

| Policy | Power (W) | EE (Mb/J) | Drop (%) | Saving (%) |
|--------|-----------|-----------|----------|------------|
| Always On | 648.56 | 0.060 | 0.00 | 0.00 |
| **Auto-SM1** | **486.12** | **0.080** | **0.00** | **25.05** |
| Simple | 545.16 | 0.072 | 0.43 | 15.94 |
| Sleepy | 565.00 | 0.069 | 6.41 | 12.89 |
| Random | 267.19 | 0.146 | 24.81 | 58.80 |

> ✅ **25% energy reduction at 0% user drop ratio** — validated over a full 24-hour urban simulation cycle.

---

## 🧠 Architecture

### How It Works

Each Base Station is an independent RL agent that:
1. **Observes** a 4-dimensional continuous state vector at every timestep
2. **Decides** which power mode to activate using a learned DQN policy
3. **Gets rewarded** for saving energy without dropping users
4. **Updates** its neural network via Experience Replay and Bellman equation backpropagation

### Core Components

| Component | Description |
|-----------|-------------|
| **Deep Q-Network (DQN)** | Multi-layer perceptron replacing tabular Q-learning to handle continuous state spaces |
| **Experience Replay Buffer** | Stores `(s, a, r, s')` tuples; random minibatch sampling breaks training correlations |
| **Target Network** | Separate frozen network for stable Bellman target computation; updated periodically |
| **Epsilon-Greedy Exploration** | Decaying ε schedule — starts near 1.0 (explore), decays to minimum (exploit) |
| **Gymnasium Environment** | Custom `gym.Env` wrapping the full 5G network simulation |
| **Multi-Agent Coordination** | Each BS agent runs independently; collective behavior optimizes network-wide energy |

---

## 📁 Repository Structure

```text
AI_IN_5G/
├── config/
│   └── settings.json          # All hyperparameters (epsilon, gamma, batch size, lr, power modes)
├── src/
│   ├── agent/
│   │   ├── dqn_agent.py       # Core DQN logic: epsilon-greedy selection, Bellman updates
│   │   ├── models.py          # PyTorch neural network architecture (policy + target networks)
│   │   └── replay_buffer.py   # Experience memory buffer for stochastic minibatch sampling
│   ├── env/
│   │   └── network_env.py     # Full 5G HetNet simulation (users, load, bandwidth, pathloss, SINR)
│   └── utils/
│       └── plotter.py         # Matplotlib + Seaborn visualization for all training metrics
├── main.py                    # Entrypoint — CLI argument parsing, training loop orchestration
├── requirements.txt           # All Python dependencies pinned
├── training.log               # Auto-generated episode-by-episode training log
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rishi12145/AI_IN_5G.git
cd AI_IN_5G
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Simulation

```bash
python main.py --config config/settings.json
```

---

## 🎛️ Configuration

All hyperparameters are controlled from `config/settings.json` — no source code changes needed:

```json
{
  "epsilon_start": 1.0,
  "epsilon_min": 0.01,
  "epsilon_decay": 0.995,
  "gamma": 0.99,
  "learning_rate": 0.001,
  "batch_size": 64,
  "replay_buffer_size": 10000,
  "target_update_freq": 10,
  "episodes": 500,
  "power_modes": {
    "deep_sleep": 2,
    "low_power_idle": 15,
    "active_transmit": 40
  },
  "qos_penalty_weight": 1.5,
  "energy_save_weight": 1.0
}
```

| Parameter | Description |
|-----------|-------------|
| `epsilon_start / min / decay` | Controls exploration-exploitation tradeoff schedule |
| `gamma` | Discount factor — how much future rewards are valued |
| `learning_rate` | Adam optimizer step size |
| `batch_size` | Minibatch size sampled from replay buffer per update |
| `replay_buffer_size` | Maximum number of stored experience tuples |
| `target_update_freq` | Episodes between target network weight syncs |
| `power_modes` | Hardware wattage for each BS power state |
| `qos_penalty_weight (β)` | Penalty multiplier for dropped users |
| `energy_save_weight (α)` | Reward multiplier for energy saved |

---

## 📊 Output Plots

After training completes, `plotter.py` automatically generates four diagnostic plots saved to the project root:

| File | Description |
|------|-------------|
| `reward_vs_episodes.png` | Cumulative reward per episode — confirms DQN convergence |
| `energy_consumption.png` | 24-hour power draw comparison across all 5 policies |
| `active_vs_sleep.png` | BS time spent in Active vs Sleep states per policy |
| `advanced_marl_metrics.png` | Q-value convergence, policy entropy decay, cooperative reward |

---

## 📊 Benchmark Policies Comparison

The benchmark evaluates the performance of the **DQN agent** against five baseline policies to measure energy efficiency and network performance.

---

### 🔹 Always On
- All base stations (BS) remain fully active **24/7**
- Results in **maximum power consumption**
- Provides **zero energy savings**
- Serves as the **upper bound baseline**

---

### 🔹 Auto-SM1
- Uses a **threshold-based heuristic**
- Transitions to **SM1 sleep mode when idle**
- Achieves **efficient energy savings with minimal performance loss**
- Considered the **best deterministic policy**

---

### 🔹 Simple
- Implements **basic rule-based sleep scheduling**
- Provides **moderate energy savings**
- May cause **occasional performance drops**

---

### 🔹 Sleepy
- Applies **aggressive sleep scheduling**
- Leads to **higher performance drops (latency/data loss)**
- Achieves **lower efficiency compared to Auto-SM1**

---

### 🔹 Random
- Selects power modes using **uniform random decisions**
- Used as a **sanity check baseline**
- Typically results in **poor and inconsistent performance**

---

## 🎯 Purpose of Benchmark
These policies provide a reference framework to evaluate how effectively the **DQN-based reinforcement learning agent**:
- Optimizes energy consumption  
- Maintains network performance  
- Adapts dynamically compared to static or heuristic approaches  


---

## 🔬 Technical Stack

| Library | Usage |
|---------|-------|
| `torch`, `torch.nn`, `torch.optim` | DQN neural networks, Adam optimizer, MSELoss |
| `gymnasium` | Standardized RL environment interface |
| `numpy` | State vector computation, stochastic mobility simulation |
| `matplotlib`, `seaborn` | Training convergence and network metric visualization |
| `json` | Global hyperparameter configuration injection |
| `argparse` | CLI execution with custom config paths |
| `logging` | Persistent per-episode training logs to `training.log` |

---

## 📄 Paper

This project is documented as a full IEEE-format conference paper:

**"Multi-Agent Deep Reinforcement Learning for Energy-Efficient 5G Networks"**  
Rishikesh Gopal, Deep Bartaria, Shreya Tiwari, Abhi Trivedi  
*Indian Institute of Information Technology Vadodara, 2025*

---

## 👥 Authors

| Name | Roll No |
|------|---------|
| Rishikesh Gopal | 202311072 |
| Deep Bartaria | 202311025 |
| Shreya Tiwari | 202311080 |
| Abhi Trivedi | 202311001 |

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.



