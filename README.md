# Finance Optimal Execution & Deep RL Project

<!-- Project illustration (add your image to the repo and update the path below) -->
![Project Overview](C:\Users\maryam\Downloads\finance-20250726T171941Z-1-001\finance\text_images\1_YzStLuSTn90DxBmm2Ncy0A.png)

This project explores optimal execution of portfolio transactions using the Almgren-Chriss model and modern deep reinforcement learning (DRL) methods.

## Contents

- **Jupyter Notebooks**: Step-by-step explanations and experiments on optimal liquidation, trading lists, efficient frontier, and DRL approaches.
- **Python Modules**: Custom environments and agent implementations for multi-agent and single-agent RL.
- **Experiments**: Scripts for training and evaluating DDPG, SAC, TD3, PPO agents, and comparison with analytical solutions.
- **Results**: Scripts and code for plotting and analyzing experiment results.

## Main Folders & Files

- `finance/`  
  - `*.ipynb` — Notebooks for theory, simulation, and DRL experiments.
  - `MultiAgent_simulation/` — Multi-agent RL environment and training scripts.
  - `syntheticChrissAlmgren.py` — Almgren-Chriss simulation environment.
  - `ddpg_agent.py`, `sac_agent.py`, `td3_agent.py`, `ppo_agent.py` — RL agent implementations.
  - `train.py`, `run_experiments.py`, `optimize.py` — Training and experiment scripts.
  - `utils.py` — Helper functions for experiments and plotting.
  - Training results for each RL agent can be accessed in the `4-DRL.ipynb` notebook.

## Getting Started

1. **Clone the repository**  
   ```
   git clone https://github.com/Mariihmp/finance.git
   cd finance
   ```

2. **Install dependencies**  
   - Python 3.8+
   - `pip install -r requirements.txt` (create this file as needed)
   - For RL experiments: `pip install torch stable-baselines3 gymnasium`

3. **Run Notebooks**  
   Open notebooks in Jupyter or VS Code for interactive exploration.

4. **Train RL Agents**  
   Example:
   ```
   python train.py --agent DDPG --episodes 5000 --risk_lambda 1e-6
   ```

## Project Highlights

- Almgren-Chriss optimal execution model.
- Efficient frontier visualization.
- Deep RL agents (DDPG, SAC, TD3, PPO) for optimal trading.
- Multi-agent market simulation.
- Experiment tracking and result analysis.

## License

MIT License

---