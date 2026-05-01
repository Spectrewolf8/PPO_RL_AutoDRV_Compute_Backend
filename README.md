# PPO RL AutoDRV - Compute Backend

Reinforcement Learning backend for autonomous driving using Proximal Policy Optimization (PPO). Trains an agent to navigate in a Unity 3D environment via ZeroMQ communication.

**Unity Game World**: [PPO_AutoDRW_Unity3d_GameWorld](https://github.com/Spectrewolf8/PPO_AutoDRW_Unity3d_GameWorld)

## Tech Stack

- **Language**: Python
- **RL Framework**: Custom PPO implementation in `src/ppo_model.py` + `src/ppo_controller.py`
- **Deep Learning**: PyTorch (required by `app.py` for training, model loading, and TensorBoard logging)
- **RL Environment**: Gymnasium-compatible environment (`src/environment.py`)
- **Communication**: ZeroMQ via `pyzmq` (REQ/REP protocol with Unity client)
- **Telemetry**: TensorBoard logs (`runs/`), file-based training logs (`logs/`)
- **Core Dependencies** (from `requirements.txt`):
  - `gymnasium`, `torchrl`, `numpy`, `pyzmq`, `tensordict`, `tqdm`

## System Requirements

- **Python 3.x** with `pip` and virtualenv/conda recommended
- **Unity client** running the game world repo (see link above)
- **GPU (optional)**: If you have CUDA-enabled PyTorch installed, training can use the GPU
- **Network**: Local ZeroMQ connection on `127.0.0.1:65432`

> Note: Install PyTorch separately if it is not resolved by your environment, then run `pip install -r requirements.txt`.

## Installation

```bash
# Clone and setup
git clone <repo-url>
cd PPO_RL_AutoDRV_Compute_Backend
pip install -r requirements.txt

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Quick Start

### Training

1. Edit `app.py`:

   ```python
   MODE = "train"
   CONFIG_FILE = "config.json"
   ```

2. Start backend:

   ```bash
   python app.py
   ```

3. Launch Unity client from [game world repo](https://github.com/Spectrewolf8/PPO_AutoDRW_Unity3d_GameWorld)

**Outputs:**

- Logs: `logs/train_<timestamp>.log`
- Checkpoints: `models/checkpoints/ppo_episode_<N>.pth` (every 50 episodes)
- Best model: `ppo_best.pth`
- Final model: `models/ppo_autodrive.pth`

### Resume Training

Edit `config.json`:

```json
"training": {
  "resume_from_checkpoint": "models/checkpoints/ppo_episode_1000.pth"
}
```

### Inference

1. Edit `app.py`:

   ```python
   MODE = "inference"
   ```

2. Edit `config.json`:

   ```json
   "inference": {
     "model_path": "ppo_best.pth"
   }
   ```

3. Run: `python app.py` + launch Unity client

## Architecture Overview

```
Unity Client (NetMQ) <--REQ/REP--> ZeroMQ Server (src/server.py)
                                     |
                                     v
                          Gymnasium Environment (src/environment.py)
                                     |
                                     v
                           PPO Controller + Model
                     (src/ppo_controller.py / src/ppo_model.py)
```

- **`app.py`** is the main entrypoint that selects **train** or **inference** mode.
- **`src/server.py`** handles ZeroMQ networking and step-by-step communication.
- **`src/environment.py`** provides a Gymnasium-style environment and observation processing.
- **`src/ppo_model.py`** defines the PPO actor/critic networks and update logic.
- **`src/ppo_controller.py`** wraps PPO action selection and policy loading.
- **`src/connection_manager.py`** manages client connection lifecycle and timeouts.

## Training Pipeline Details

- **Step processing**: Each Unity `game_state` message updates the environment state.
- **Transition storage**: `(obs, action, reward, done)` is stored per step.
- **Policy updates**: PPO update runs every `update_frequency` steps.
- **Checkpoints**: Saved every `save_frequency` episodes.
- **Best model**: `ppo_best.pth` saved when episode reward improves.
- **Final model**: `models/ppo_autodrive.pth` written at training completion.
- **TensorBoard**: Logs actor/critic/entropy losses into `runs/`.

## Configuration

Main settings in `config.json`:

**Server:**

- `host`: `127.0.0.1` (localhost)
- `port`: `65432` (ZeroMQ connection)
- `tickrate`: Updates per second

**Environment:**

- `max_ray_distances`: Ray sensor max distances
- `max_speed`: Vehicle max speed
- `reward_collected_value`: Reward for collectibles
- `collision_penalty`: Collision penalty
- `survival_reward`: Per-step reward
- `straight_driving_reward`: Bonus for straight driving

**Training:**

- `total_episodes`: Training episode count
- `update_frequency`: Policy update interval
- `save_frequency`: Checkpoint save interval
- `resume_from_checkpoint`: Path to resume from (or `null`)

**PPO:**

- `lr_actor`, `lr_critic`: Learning rates
- `gamma`: Discount factor (0.99)
- `epsilon`: PPO clip parameter (0.2)
- `entropy_coef`: Exploration bonus
- `batch_size`: Training batch size

## Environment Details

**Observation Space (11D):**

- 5 ray distances (normalized)
- 5 ray hit indicators (binary)
- 1 speed value

**Action Space (3 discrete):**

- 0: Turn Left
- 1: Straight
- 2: Turn Right

**Rewards:**

- Survival: +0.1/step
- Straight driving: +0.05
- Collection: +15.0
- Collision: -10.0

## Model Architecture

**Actor (Policy):** Input(11) → FC(256) → ReLU → FC(256) → ReLU → FC(3) → Softmax

**Critic (Value):** Input(11) → FC(256) → ReLU → FC(256) → ReLU → FC(1)

## Project Structure

```
PPO_RL_AutoDRV_Compute_Backend/
├── app.py                      # Main entry point
├── config.json                 # Configuration
├── requirements.txt            # Dependencies
├── src/
│   ├── server.py              # ZeroMQ server
│   ├── environment.py         # Gym environment
│   ├── ppo_model.py           # PPO algorithm
│   ├── ppo_controller.py      # Agent controller
│   ├── connection_manager.py  # Connection handling
│   └── helpers.py             # Utilities
├── models/
│   ├── ppo_autodrive.pth      # Final model
│   ├── ppo_best.pth           # Best model
│   └── checkpoints/           # Training checkpoints
└── logs/                       # Training logs
```

## Troubleshooting

**Connection Issues:**

- Verify Unity and Python use same `host:port`
- Check firewall settings

**Training Issues:**

- Lower learning rates if unstable
- Adjust reward structure
- Increase `collision_penalty` if too aggressive

**GPU Not Working:**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

**Communication Protocol:** See [CommunicationDesign.md](CommunicationDesign.md) for ZeroMQ protocol details.