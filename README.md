# PPO RL AutoDRV - Compute Backend

Reinforcement Learning backend for autonomous driving using Proximal Policy Optimization (PPO). This system trains an agent to navigate autonomously in a Unity-based driving environment.

## Architecture Overview

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│                 │         │                  │         │                 │
│  Unity Game     │ ◄─────► │  Communication   │ ◄─────► │  Gymnasium      │
│  (Client)       │  ZeroMQ │  Server          │         │  Environment    │
│                 │         │  (server.py)     │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                     │                            │
                                     │                            │
                                     ▼                            ▼
                            ┌──────────────────┐         ┌─────────────────┐
                            │  PPO Controller  │         │  AutoDrivingEnv │
                            │                  │         │                 │
                            └──────────────────┘         └─────────────────┘
                                     │
                                     │
                                     ▼
                            ┌──────────────────┐
                            │   PPO Model      │
                            │  (Actor-Critic)  │
                            └──────────────────┘
```

- **app.py**: Main entry point (training/inference modes)
- **server.py**: Communication bridge between Unity and environment
- **environment.py**: Gymnasium environment for autonomous driving
- **ppo_model.py**: PPO algorithm implementation (Actor-Critic)
- **ppo_controller.py**: Controller for PPO agent interaction

## Features

- ✅ **Training Mode**: Train PPO agent from scratch
- ✅ **Resume Training**: Continue training from saved checkpoints
- ✅ **Inference Mode**: Run trained models for testing/deployment
- ✅ **Configurable**: JSON-based configuration for all parameters
- ✅ **Checkpoint System**: Automatic model saving during training with full state preservation
- ✅ **Real-time Communication**: ZeroMQ bridge to Unity
- ✅ **Flexible Rewards**: Configurable reward structure with straight driving bonus
- ✅ **GPU Support**: Automatic CUDA detection

## Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd PPO_RL_AutoDRV_Compute_Backend
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Verify installation**

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Quick Start

### Training a New Model

1. **Configure training** (edit config.json if needed - defaults are provided)

2. **Set mode in app.py**:

```python
MODE = "train"              # Set to "train"
CONFIG_FILE = "config.json"  # Your config file
```

3. **Start training server**:

```bash
python app.py
```

4. **Launch Unity client** and connect to `127.0.0.1:65432`

5. **Monitor training**:
   - Logs: `logs/train_<timestamp>.log`
   - Checkpoints: `models/checkpoints/`
   - Best model: ppo_best.pth
   - Final model: `models/ppo_autodrive.pth`

### Resuming Training from Checkpoint

1. **Find your checkpoint** (e.g., `models/checkpoints/ppo_episode_50.pth`)

2. **Update config.json**:

```json
"training": {
  "resume_from_checkpoint": "models/checkpoints/ppo_episode_50.pth"
}
```

3. **Start training server**:

```bash
python app.py
```

4. **Training will resume** from the saved episode with all progress preserved:
   - Episode count
   - Training steps
   - Best reward achieved
   - Model weights and optimizer states

### Running Inference

1. **Ensure your trained model exists** (e.g., ppo_best.pth or `models/ppo_autodrive.pth`)

2. **Set mode in app.py**:

```python
MODE = "inference"          # Set to "inference"
CONFIG_FILE = "config.json"  # Your config file
```

3. **Start inference server**:

```bash
python app.py
```

4. **Launch Unity client** to see the trained agent in action

## Configuration

### config.json Structure

**Note:** The actual config.json file should NOT contain comments (JSON doesn't support them). The comments below are for documentation purposes only.

**Mode Selection:** Training vs Inference mode is set in app.py via the `MODE` variable, NOT in the config file.

```javascript
{
  "server": {
    "host": "127.0.0.1",        // Server host address
    "port": 65432,               // Server port number
    "tickrate": 30               // Updates per second
  },

  "environment": {
    "max_ray_distances": [7.0, 4.5, 4.5, 3.5, 3.5],  // Max distances for each ray sensor
                                                        // [Forward, Fwd-Left, Fwd-Right, Right, Left]
    "max_speed": 2.5,                                 // Maximum car speed
    "steering_speed_penalty": 0.5,                    // Speed reduction when steering (handled by Unity)
    "reward_collected_value": 15.0,                   // Reward for collecting items
    "collision_penalty": -10.0,                       // Penalty for collisions
    "survival_reward": 0.1,                           // Small reward per step alive
    "straight_driving_reward": 0.05,                  // Reward for driving straight (encourages stability)
    "max_episode_steps": 1000                         // Max steps per episode before truncation
  },

  "training": {
    "total_episodes": 1000,                           // Number of episodes to train
    "update_frequency": 200,                          // Update policy every N steps
    "save_frequency": 50,                             // Save checkpoint every N episodes
    "model_save_path": "models/ppo_autodrive.pth",   // Final model save path
    "checkpoint_dir": "models/checkpoints",           // Directory for training checkpoints
    "log_dir": "logs",                                // Directory for log files
    "resume_from_checkpoint": null                    // Path to checkpoint to resume from (or null for new training)
  },

  "ppo_hyperparameters": {
    "lr_actor": 0.0003,                               // Actor (policy) learning rate
    "lr_critic": 0.001,                               // Critic (value) learning rate
    "gamma": 0.99,                                    // Discount factor for future rewards
    "gae_lambda": 0.95,                               // GAE (Generalized Advantage Estimation) lambda
    "epsilon": 0.2,                                   // PPO clip parameter
    "entropy_coef": 0.01,                             // Entropy bonus coefficient for exploration
    "value_coef": 0.5,                                // Value loss coefficient
    "max_grad_norm": 0.5,                             // Gradient clipping threshold
    "hidden_dim": 256,                                // Hidden layer dimension for neural networks
    "update_epochs": 10,                              // Number of epochs per policy update
    "batch_size": 64                                  // Batch size for policy updates
  },

  "inference": {
    "model_path": "models/ppo_autodrive.pth",        // Path to trained model for inference
    "deterministic": true                             // Use deterministic actions (no exploration)
  }
}
```

## Usage

### Switching Modes

**Important:** The operation mode (training vs inference) is controlled in app.py, NOT in the config file.

Edit the flags at the top of app.py:

```python
# CONFIGURATION FLAGS - Edit these to change behavior
MODE = "train"              # "train" or "inference"
CONFIG_FILE = "config.json"  # Path to configuration file
```

### Running the Application

```bash
# Always run with:
python app.py
```

### Quick Mode Switching

**For Training:**

```python
MODE = "train"
CONFIG_FILE = "config.json"
```

**For Inference:**

```python
MODE = "inference"
CONFIG_FILE = "config.json"
```

**For Quick Testing:**

```python
MODE = "train"
CONFIG_FILE = "config_quicktest.json"
```

## Environment Details

### Observation Space (11 dimensions)

- **5 ray distances**: Normalized distances from ray sensors
  - Forward, Forward-Left, Forward-Right, Right, Left
- **5 ray hit indicators**: Binary (0 or 1) if ray hit something
- **1 speed value**: Current car speed (0 to max_speed)

### Action Space (Discrete 3)

- **0**: Turn Left (-1 steering)
- **1**: Straight (0 steering)
- **2**: Turn Right (+1 steering)

### Reward Structure

- `+survival_reward`: Per step alive (default: +0.1)
- `+straight_driving_reward`: For driving straight (default: +0.05)
- `+reward_collected_value`: For collecting items (default: +15.0)
- `+collision_penalty`: For collisions (default: -10.0)

**Note:** The straight driving reward encourages the agent to prefer stable, straight-line driving when it's safe to do so, promoting more efficient navigation.

## Model Architecture

### Actor Network (Policy)

```
Input (11) → FC(256) → ReLU → FC(256) → ReLU → FC(3) → Softmax
```

### Critic Network (Value Function)

```
Input (11) → FC(256) → ReLU → FC(256) → ReLU → FC(1)
```

### Hyperparameters

- **State Dimension**: 11
- **Action Dimension**: 3 (discrete)
- **Hidden Dimension**: 256
- **Optimizer**: Adam
- **Learning Rate**: 3e-4 (actor), 1e-3 (critic)
- **Discount Factor (γ)**: 0.99
- **GAE Lambda (λ)**: 0.95
- **Clip Parameter (ε)**: 0.2

## Training Tips

### Resume Training

**When to Resume:**

- Training was interrupted (Ctrl+C, power loss, etc.)
- Want to extend training beyond original episode count
- Need to fine-tune an existing model with different hyperparameters

**How to Resume:**

1. **Locate your checkpoint:**
   - Periodic: `models/checkpoints/ppo_episode_<N>.pth`
   - Best: ppo_best.pth
   - Interrupted: `models/checkpoints/ppo_interrupted_ep<N>.pth`

2. **Update config.json:**

   ```json
   "training": {
     "total_episodes": 1500,  // Extend beyond original 1000
     "resume_from_checkpoint": "models/checkpoints/ppo_episode_1000.pth"
   }
   ```

3. **Run training:**
   ```bash
   python app.py
   ```

**What Gets Preserved:**

- ✅ Model weights (Actor & Critic networks)
- ✅ Optimizer states (Adam momentum, etc.)
- ✅ Current episode number
- ✅ Total training steps
- ✅ Best episode reward achieved
- ✅ All hyperparameters

**Tips:**

- To start fresh, set `resume_from_checkpoint: null`
- Checkpoints include full training state for seamless continuation
- Best model is automatically tracked and saved

### For Better Performance

1. **Adjust learning rates**: Lower for stable learning, higher for faster convergence
2. **Tune reward structure**: Balance survival, collection, collision, and straight driving rewards
3. **Modify update frequency**: More frequent updates = more stable but slower
4. **Entropy coefficient**: Higher = more exploration, lower = more exploitation
5. **Straight driving reward**: Adjust to control preference for stable vs. aggressive driving

### Monitoring Training

- Watch episode rewards increasing over time
- Check loss values (should decrease)
- Monitor entropy (should decrease gradually)
- Review checkpoint performance

### Common Issues

- **Agent not learning**: Increase learning rate or entropy coefficient
- **Agent too cautious**: Reduce collision penalty
- **Agent too aggressive**: Increase collision penalty
- **Training unstable**: Reduce learning rates, increase batch size

## Directory Structure

```
PPO_RL_AutoDRV_Compute_Backend/
├── app.py                      # Main entry point (THIS FILE)
├── config.json                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Source code
│   ├── server.py              # Communication server
│   ├── environment.py         # Gymnasium environment
│   ├── ppo_model.py           # PPO algorithm
│   ├── ppo_controller.py      # Agent controller
│   ├── connection_manager.py  # ZeroMQ connection handler
│   └── helpers.py             # Utility functions
│
├── models/                     # Trained models
│   ├── ppo_autodrive.pth      # Final trained model
│   ├── ppo_best.pth           # Best model during training
│   └── checkpoints/           # Training checkpoints
│
└── logs/                       # Training/inference logs
    ├── train_<timestamp>.log
    └── inference_<timestamp>.log
```

## API Reference

### app.py

#### Configuration Flags

Edit these at the top of app.py:

```python
MODE = "train"              # "train" or "inference"
CONFIG_FILE = "config.json"  # Path to configuration file
```

#### Training Mode

Set `MODE = "train"` and run `python app.py`

- Starts training server
- Connects to Unity client
- Trains PPO agent through environment interactions
- Saves checkpoints and best model
- Logs training progress

#### Inference Mode

Set `MODE = "inference"` and run `python app.py`

- Loads trained model
- Starts inference server
- Runs agent in Unity environment
- Logs performance metrics

### Configuration Loading

```python
from app import load_config
config = load_config("config.json")
```

## Advanced Usage

### Custom Training Loop

```python
from src.environment import AutoDrivingEnv
from src.ppo_model import PPO

env = AutoDrivingEnv()
ppo = PPO(state_dim=11, action_dim=3)

for episode in range(1000):
    state, _ = env.reset()
    done = False

    while not done:
        action = ppo.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        ppo.store_transition(state, action, reward, terminated or truncated)
        state = next_state
        done = terminated or truncated

    # Update policy
    losses = ppo.update()

    # Save periodically
    if episode % 50 == 0:
        ppo.save(f"models/checkpoint_{episode}.pth")
```

### Custom Reward Function

Edit environment.py or configure in config.json:

```python
env = AutoDrivingEnv(
    reward_collected_value=20.0,      # Increase collection reward
    collision_penalty=-15.0,          # Increase collision penalty
    survival_reward=0.2,              # Increase survival reward
    straight_driving_reward=0.1       # Increase straight driving bonus
)
```

## Troubleshooting

### Connection Issues

- Ensure Unity client and server use same host/port
- Check firewall settings
- Verify ZeroMQ is installed: `pip install pyzmq`

### Training Not Converging

- Lower learning rates
- Increase batch size
- Adjust reward structure
- Check environment is providing meaningful rewards

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

- Reduce batch size in config
- Reduce network hidden dimensions
- Use CPU instead of GPU
