"""
PPO RL AutoDRV - Main Application Entry Point

This script serves as the main entrypoint for the PPO-based autonomous driving system.
It supports two modes:
  - Training Mode: Train a PPO agent using the Unity environment
  - Inference Mode: Run a trained PPO model for inference/testing

The server (src/server.py) acts as a communication bridge between the Unity game
and the Gymnasium environment. The environment interactions naturally affect model
training through the PPO algorithm.

Usage:
  1. Set MODE and CONFIG_FILE variables below
  2. Run: python app.py

Configuration is read from a JSON file (default: config.json)
"""

import sys
import os
import json
import logging
from datetime import datetime
import torch.utils.tensorboard as tb


# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from environment import AutoDrivingEnv
from ppo_controller import PPOController
from server import GameServer


# Configure logging
def setup_logging(log_dir: str = "logs", mode: str = "train") -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{mode}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"✓ Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"✗ Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)


def create_environment(env_config: dict) -> AutoDrivingEnv:
    """Create and configure the AutoDriving environment."""
    env = AutoDrivingEnv(
        max_ray_distances=env_config.get("max_ray_distances", [7.0, 4.5, 4.5, 3.5, 3.5]),
        max_speed=env_config.get("max_speed", 2.5),
        steering_speed_penalty=env_config.get("steering_speed_penalty", 0.5),
        reward_collected_value=env_config.get("reward_collected_value", 15.0),
        collision_penalty=env_config.get("collision_penalty", -10.0),
        survival_reward=env_config.get("survival_reward", 0.1),
        straight_driving_reward=env_config.get("straight_driving_reward", 0.05),
    )
    env._max_episode_steps = env_config.get("max_episode_steps", 1000)
    return env


class TrainingServer(GameServer):
    """Extended GameServer with training capabilities."""

    def __init__(self, config: dict, logger: logging.Logger):
        """Initialize training server."""
        self.config = config
        self.app_logger = logger

        # Extract configurations
        server_config = config.get("server", {})
        env_config = config.get("environment", {})
        train_config = config.get("training", {})
        ppo_config = config.get("ppo_hyperparameters", {})

        # Initialize base server (without model for training)
        super().__init__(
            host=server_config.get("host", "127.0.0.1"),
            port=server_config.get("port", 65432),
            tickrate=server_config.get("tickrate", 30),
            model_path=None,  # No pre-trained model for training
        )

        # Override environment with configured values
        self.env = create_environment(env_config)
        self.controller = PPOController(self.env)

        # Training configuration
        self.total_episodes = train_config.get("total_episodes", 1000)
        self.update_frequency = train_config.get("update_frequency", 200)
        self.save_frequency = train_config.get("save_frequency", 50)
        self.model_save_path = train_config.get("model_save_path", "models/ppo_autodrive.pth")
        self.checkpoint_dir = train_config.get("checkpoint_dir", "models/checkpoints")
        resume_checkpoint = train_config.get("resume_from_checkpoint", None)

        # Training state
        self.training_steps = 0
        self.last_update_step = 0
        self.best_episode_reward = float("-inf")

        # Create directories
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Add TensorBoard writer
        # Setup TensorBoard logging with readable timestamp
        tensorboard_log_dir = f"runs/training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.writer = tb.SummaryWriter(log_dir=tensorboard_log_dir)
        self.app_logger.info("TensorBoard logs: runs/")

        # Create or load PPO model
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            self.app_logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            self.ppo_model = self.controller.create_model(**ppo_config)
            training_state = self.ppo_model.load(resume_checkpoint)

            if training_state:
                # Resume from saved training state
                self.current_episode = training_state.get("current_episode", 0)
                self.training_steps = training_state.get("training_steps", 0)
                self.last_update_step = training_state.get("last_update_step", 0)
                self.best_episode_reward = training_state.get("best_episode_reward", float("-inf"))

                self.app_logger.info("=" * 70)
                self.app_logger.info("RESUMING TRAINING FROM CHECKPOINT")
                self.app_logger.info("=" * 70)
                self.app_logger.info(f"Resuming from Episode: {self.current_episode}")
                self.app_logger.info(f"Training Steps: {self.training_steps}")
                self.app_logger.info(f"Best Episode Reward: {self.best_episode_reward:.2f}")
                self.app_logger.info(f"Episodes Remaining: {self.total_episodes - self.current_episode}")
                self.app_logger.info("=" * 70)
            else:
                self.app_logger.warning("Checkpoint loaded but no training state found. Starting fresh.")
        else:
            if resume_checkpoint:
                self.app_logger.warning(f"Checkpoint not found: {resume_checkpoint}. Starting new training.")
            self.ppo_model = self.controller.create_model(**ppo_config)

        self.app_logger.info("=" * 70)
        self.app_logger.info("TRAINING MODE INITIALIZED")
        self.app_logger.info("=" * 70)
        self.app_logger.info(f"Total Episodes: {self.total_episodes}")
        self.app_logger.info(f"Update Frequency: {self.update_frequency} steps")
        self.app_logger.info(f"Save Frequency: {self.save_frequency} episodes")
        self.app_logger.info(f"Model Save Path: {self.model_save_path}")
        self.app_logger.info("=" * 70)

    def _process_game_state(self, game_state: dict) -> dict:
        """Override to add training logic."""
        # Get base response from parent
        response = super()._process_game_state(game_state)

        # Extract training data
        state_data = game_state.get("gameState", {})
        self.env.update_state(state_data)

        # Get current observation (as numpy array, not dict)
        observation = self.env._get_observation()
        action = response.get("action", 1)  # Get action from controller
        reward = response.get("reward", 0.0)
        terminated = response.get("terminated", False)

        # Store transition in PPO memory
        self.ppo_model.store_transition(observation, action, reward, terminated)
        self.training_steps += 1

        # Update policy at specified frequency
        if self.training_steps - self.last_update_step >= self.update_frequency:
            losses = self.ppo_model.update()

            # Log to TensorBoard
            self.writer.add_scalar("Loss/Actor", losses["actor_loss"], self.training_steps)
            self.writer.add_scalar("Loss/Critic", losses["critic_loss"], self.training_steps)
            self.writer.add_scalar("Loss/Entropy", losses["entropy"], self.training_steps)

            self.app_logger.info(
                f"[TRAINING] Step {self.training_steps} - "
                f"Actor Loss: {losses['actor_loss']:.4f}, "
                f"Critic Loss: {losses['critic_loss']:.4f}, "
                f"Entropy: {losses['entropy']:.4f}"
            )
            self.last_update_step = self.training_steps

        return response

    def _handle_episode_end(self) -> None:
        """Override to add model saving logic."""
        # Call parent to log statistics
        super()._handle_episode_end()

        # Save model periodically
        if self.current_episode % self.save_frequency == 0 and self.current_episode > 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"ppo_episode_{self.current_episode}.pth")
            training_state = {
                "current_episode": self.current_episode,
                "training_steps": self.training_steps,
                "last_update_step": self.last_update_step,
                "best_episode_reward": self.best_episode_reward,
            }
            self.ppo_model.save(checkpoint_path, training_state=training_state)
            self.app_logger.info(f"[TRAINING] Checkpoint saved: {checkpoint_path}")

        # Save best model
        if self.episode_reward > self.best_episode_reward:
            self.best_episode_reward = self.episode_reward
            best_model_path = os.path.join(os.path.dirname(self.model_save_path), "ppo_best.pth")
            training_state = {
                "current_episode": self.current_episode,
                "training_steps": self.training_steps,
                "last_update_step": self.last_update_step,
                "best_episode_reward": self.best_episode_reward,
            }
            self.ppo_model.save(best_model_path, training_state=training_state)
            self.app_logger.info(
                f"[TRAINING] New best model! Reward: {self.best_episode_reward:.2f} " f"saved to {best_model_path}"
            )

        # Check if training is complete
        if self.total_episodes_completed >= self.total_episodes:
            self.app_logger.info("=" * 70)
            self.app_logger.info("TRAINING COMPLETE!")
            self.app_logger.info(f"Total Episodes: {self.total_episodes_completed}")
            self.app_logger.info(f"Total Steps: {self.training_steps}")
            self.app_logger.info(f"Best Episode Reward: {self.best_episode_reward:.2f}")
            self.app_logger.info("=" * 70)

            # Save final model
            training_state = {
                "current_episode": self.current_episode,
                "training_steps": self.training_steps,
                "last_update_step": self.last_update_step,
                "best_episode_reward": self.best_episode_reward,
            }
            self.ppo_model.save(self.model_save_path, training_state=training_state)
            self.app_logger.info(f"Final model saved to: {self.model_save_path}")

            # Shutdown server
            self.shutdown()

    @property
    def total_episodes_completed(self) -> int:
        """Get total completed episodes."""
        return self.current_episode


def run_training_mode(config: dict, logger: logging.Logger):
    """Run the application in training mode."""
    logger.info("Starting Training Mode...")

    server = TrainingServer(config, logger)

    try:
        logger.info("Training server starting. Waiting for Unity client connection...")
        server.start()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

        # Save current model state
        current_model_path = os.path.join(
            config["training"]["checkpoint_dir"], f"ppo_interrupted_ep{server.current_episode}.pth"
        )
        training_state = {
            "current_episode": server.current_episode,
            "training_steps": server.training_steps,
            "last_update_step": server.last_update_step,
            "best_episode_reward": server.best_episode_reward,
        }
        server.ppo_model.save(current_model_path, training_state=training_state)
        logger.info(f"Model saved before exit: {current_model_path}")

        server.shutdown()
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)

        # Save model before shutdown
        try:
            emergency_path = os.path.join(
                config["training"]["checkpoint_dir"], f"ppo_emergency_ep{server.current_episode}.pth"
            )
            training_state = {
                "current_episode": server.current_episode,
                "training_steps": server.training_steps,
                "last_update_step": server.last_update_step,
                "best_episode_reward": server.best_episode_reward,
            }
            server.ppo_model.save(emergency_path, training_state=training_state)
            logger.info(f"Emergency save completed: {emergency_path}")
        except Exception as save_error:
            logger.error(f"Failed to save model during emergency shutdown: {save_error}")

        server.shutdown()


def run_inference_mode(config: dict, logger: logging.Logger):
    """Run the application in inference mode."""
    logger.info("Starting Inference Mode...")

    # Extract configurations
    server_config = config.get("server", {})
    env_config = config.get("environment", {})
    inference_config = config.get("inference", {})

    # Get model path
    model_path = inference_config.get("model_path", "models/ppo_autodrive.pth")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please train a model first or provide a valid model path in config.")
        sys.exit(1)

    logger.info(f"Loading trained model from: {model_path}")

    # Create server with trained model
    server = GameServer(
        host=server_config.get("host", "127.0.0.1"),
        port=server_config.get("port", 65432),
        tickrate=server_config.get("tickrate", 30),
        model_path=model_path,
    )

    # Override environment with configured values
    server.env = create_environment(env_config)
    server.controller = PPOController(server.env)
    server.controller.load_model(model_path)

    logger.info("=" * 70)
    logger.info("INFERENCE MODE INITIALIZED")
    logger.info("=" * 70)
    logger.info(f"Model Path: {model_path}")
    logger.info(f"Deterministic: {inference_config.get('deterministic', True)}")
    logger.info("=" * 70)

    try:
        logger.info("Inference server starting. Waiting for Unity client connection...")
        server.start()
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
        server.shutdown()
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        server.shutdown()


def main():
    """Main entry point."""

    # CONFIGURATION FLAGS - Edit these to change behavior
    MODE = "train"  # "train" or "inference"
    CONFIG_FILE = "config_quicktest.json"  # Path to configuration file

    # Load configuration
    config = load_config(CONFIG_FILE)

    # Use mode from app.py (not from config file)
    mode = MODE

    # Validate mode
    if mode not in ["train", "inference"]:
        print(f"✗ Error: Invalid mode '{mode}'. Must be 'train' or 'inference'.")
        print("Edit the MODE variable at the top of app.py")
        sys.exit(1)

    # Setup logging
    log_dir = config.get("training", {}).get("log_dir", "logs")
    logger = setup_logging(log_dir, mode)

    # Display banner
    print("\n" + "=" * 70)
    print("       PPO RL AutoDRV - Autonomous Driving System")
    print("=" * 70)
    print(f"Mode:   {mode.upper()}")
    print(f"Config: {CONFIG_FILE}")
    print("=" * 70 + "\n")

    # Run appropriate mode
    if mode == "train":
        run_training_mode(config, logger)
    elif mode == "inference":
        run_inference_mode(config, logger)


if __name__ == "__main__":
    main()
