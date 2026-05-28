# Congestion-Aware HRL Warehouse Fulfillment

This project is a grid-based multi-robot warehouse fulfillment simulator using hierarchical reinforcement learning. The high-level RL scheduler assigns robots to customer orders, while A* path planning handles low-level robot navigation around shelves and obstacles.

The project compares the learned HRL scheduler against greedy, congestion-aware greedy, and random baseline policies using delivery throughput, reward, and collision/congestion metrics.

## Project Structure

```text
config.py                  Environment size, reward values, and congestion settings
warehouse_env.py           Gymnasium warehouse environment
controllers.py             Greedy, congestion-aware greedy, and random schedulers
astar.py                   A* path planner for robot navigation
train_hrl_scheduler.py     PPO training script for the HRL scheduler
evaluate.py                Evaluation script for baselines and HRL
demo_heuristic.py          Greedy rollout demo
requirements.txt           Required Python packages
outputs/                   Generated model, charts and CSV files
```

## Environment Setup

The final experiments were run in the following warehouse configuration:

```text
Grid size:          14 x 10
Number of robots:  3
Maximum orders:    5
Maximum steps:     300 per episode
Observation shape: 52
Action space:      Discrete(16)
```

Each HRL action represents a high-level robot-order assignment or a no-op action. A* is used for low-level path planning after an assignment is made.

## How It Works

The system is hierarchical:

```text
HRL scheduler
    chooses robot-order assignments
        ↓
A* path planner
    computes obstacle-free paths
        ↓
Warehouse simulator
    updates robot movement, deliveries, rewards, and collisions
```

## How to Run

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the HRL scheduler:

```bash
py train_hrl_scheduler.py 
```

Evaluate all policies:

```bash
py evaluate.py
```


## Results

The final comparison across the main tested variants is shown below.

| Policy | Average Reward | Average Deliveries | Average Collision/Congestion Events | Model Description|
|---|---:|---:|---:|---|
| Greedy | 27.37 | 2.0 | 832.0 | Nearest-task rule baseline |
| HRL-Base | 160.29 | 4.3 | 769.6 | PPO HRL without congestion reward |
| HRL-Congestion | 246.04 | 5.9 | 746.1 | PPO HRL with A*-based path-overlap penalty |
| HRL-Actor-BC | **303.25** | **6.7** | 743.2 | A*-guided actor pretraining followed by PPO |
| HRL-Property-Critic | 225.87 | 6.4 | **734.5** | Actor + system-property critic pretraining |


The earlier HRL setup could get close to greedy delivery performance, but collision reduction was limited because the scheduler only received collision penalties after conflicts happened. In the final congestion-aware version, A* is also used to estimate path-overlap risk before assignment. This gave the HRL policy a more useful signal for avoiding crowded routes.

The best overall result came from **HRL-Actor-BC**, where the actor was first pretrained using an A*-guided expert heuristic and then fine-tuned with PPO. The **HRL-Property-Critic** variant achieved the lowest collision/congestion count, but with lower reward than the actor-pretrained model.

## Findings

The main finding is that HRL became useful when the scheduler received congestion-aware information. A basic greedy scheduler only selects short-distance assignments, while the HRL policy learned from delivery rewards, collision penalties, and path-overlap risk.

The results also show that pretraining can help, but the type of pretraining matters. A*-guided actor pretraining gave the best overall reward and delivery performance, while property-critic pretraining produced the safest collision result. Reward-aligned critic pretraining improved over the safety-only critic in reward, but did not outperform actor-only pretraining.

This suggests that high-level learning is more effective when the environment provides meaningful decision features instead of relying only on delayed collision penalties.
