# 🚀 Lunar Lander PPO Agent

A reinforcement learning project that trains an autonomous agent to successfully land a spacecraft in the **Gymnasium LunarLander-v3** environment using **Proximal Policy Optimization (PPO)**. The project uses **Stable-Baselines3** for training and exports the trained policy as a lightweight NumPy parameter file for fast inference.

---

## 📌 Project Overview

The objective is to train an AI agent capable of landing a lunar spacecraft safely while maximizing cumulative reward. The trained policy is evaluated over multiple episodes using a custom evaluation script.

This project was developed as part of the **CS236 – Artificial Intelligence** course at the **Indian Institute of Information Technology (IIIT) Guwahati**.

---

## ✨ Features

- PPO-based reinforcement learning agent
- Two-hidden-layer neural network (8 → 128 → 128 → 4)
- Trained using Stable-Baselines3
- NumPy-only inference (no PyTorch required during evaluation)
- Custom evaluation script for benchmarking
- Human-play script to interact with the environment
- Lightweight exported policy parameters

---

## 🧠 Neural Network Architecture

```
Input (8)
      │
      ▼
Hidden Layer (128, ReLU)
      │
      ▼
Hidden Layer (128, ReLU)
      │
      ▼
Output Layer (4 Actions)
```

Actions:
- Do Nothing
- Fire Left Engine
- Fire Main Engine
- Fire Right Engine

---

## 📂 Project Structure

```
.
├── best_policy_2209.npy      # Trained policy parameters
├── policy_2209.py            # NumPy inference policy
├── train_agent_2209.py       # PPO training script
├── evaluate_agent.py         # Evaluation script
├── play_lunar_lander.py      # Play the game manually
├── evaluate.bat              # Evaluation command
├── methodology.pdf           # Project report
└── README.md
```

---

## 🛠️ Technologies Used

- Python 3
- Gymnasium
- Stable-Baselines3
- PyTorch
- NumPy

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/nexon728/lunar-lander-ppo-agent.git
cd lunar-lander-ppo-agent
```

Install dependencies:

```bash
pip install gymnasium[box2d] stable-baselines3 torch numpy pygame
```

---

## 🎮 Play the Environment

```bash
python play_lunar_lander.py
```

Controls:

| Key | Action |
|-----|--------|
| W | Main Engine |
| A | Left Engine |
| D | Right Engine |
| S | Do Nothing |

---

## 🏋️ Train the Agent

```bash
python train_agent_2209.py
```

Training configuration:

- Algorithm: PPO
- Parallel Environments: 32
- Total Timesteps: 10,000,000
- Hidden Layers: 128 × 128
- Activation: ReLU

---

## 📊 Evaluate the Agent

```bash
python evaluate_agent.py --filename best_policy_2209.npy --policy_module policy_2209
```

The evaluation computes the average reward over 100 episodes.

---

## 📈 Performance

| Metric | Value |
|---------|-------|
| Algorithm | PPO |
| Observation Size | 8 |
| Action Space | 4 |
| Hidden Layers | 128 → 128 |
| Parameters | 18,180 |
| Training Timesteps | 10 Million |
| Parallel Environments | 32 |

---

## 📄 Documentation

A detailed explanation of the methodology, neural network architecture, training procedure, and evaluation process is available in:

- **methodology.pdf**

---

## 🎓 Academic Information

**Course:** CS236 – Artificial Intelligence

**Institution:** Indian Institute of Information Technology (IIIT) Guwahati

---

## 📜 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**Vighnesh**

GitHub: https://github.com/nexon728