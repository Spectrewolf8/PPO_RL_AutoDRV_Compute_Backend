from typing import Dict, Any, Tuple
from environment import AutoDrivingEnv
from ppo_model import PPO
import os
import logging
import torch

logger = logging.getLogger(__name__)


class PPOController:
    """
    Controls the PPO agent interaction with the environment.
    Generates actions based on observations.
    """

    def __init__(self, env: AutoDrivingEnv, device: str = None):
        """
        Initialize the PPO controller.

        Args:
            env: AutoDrivingEnv instance
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.env = env
        self.ppo_model = None  # Will be PPO instance when loaded
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Log GPU availability
        if self.device == "cuda":
            logger.info(f"GPU ENABLED - Using {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.info("Running on CPU (GPU not available or not requested)")

    def load_model(self, model_path: str) -> None:
        """
        Load a trained PPO model.

        Args:
            model_path: Path to the saved model
        """
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}. Using rule-based policy.")
                return

            logger.info(f"Loading PPO model from {model_path}...")

            # Initialize PPO model with correct dimensions
            self.ppo_model = PPO(
                state_dim=11,  # AutoDrivingEnv observation space
                action_dim=3,  # Discrete actions: left, straight, right
                device=self.device,
            )

            # Load trained weights
            self.ppo_model.load(model_path)

            # Get model info
            actor_params = sum(p.numel() for p in self.ppo_model.actor.parameters())
            critic_params = sum(p.numel() for p in self.ppo_model.critic.parameters())

            logger.info("  PPO model loaded successfully")
            logger.info(f"  Device: {self.ppo_model.device}")
            logger.info(f"  Actor parameters: {actor_params:,}")
            logger.info(f"  Critic parameters: {critic_params:,}")
            logger.info(f"  Total parameters: {actor_params + critic_params:,}")

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            logger.warning("Falling back to rule-based policy")
            self.ppo_model = None

    def create_model(self, **kwargs) -> PPO:
        """
        Create a new PPO model for training.

        Args:
            **kwargs: PPO hyperparameters

        Returns:
            Initialized PPO model
        """
        logger.info("Creating new PPO model for training...")

        self.ppo_model = PPO(state_dim=11, action_dim=3, device=self.device, **kwargs)

        actor_params = sum(p.numel() for p in self.ppo_model.actor.parameters())
        critic_params = sum(p.numel() for p in self.ppo_model.critic.parameters())

        logger.info("  PPO model created")
        logger.info(f"  Device: {self.ppo_model.device}")
        logger.info(f"  Actor parameters: {actor_params:,}")
        logger.info(f"  Critic parameters: {critic_params:,}")
        logger.info(f"  Total parameters: {actor_params + critic_params:,}")

        return self.ppo_model

    def get_action(self, game_state: Dict[str, Any]) -> Tuple[int, int]:
        """
        Get action from PPO model or rule-based policy.

        Args:
            game_state: Dictionary containing game state from Unity

        Returns:
            Tuple of (action_index, steering_value)
        """
        # Update environment with game state
        self.env.update_state(game_state)

        # Get observation
        observation = self.env._get_observation()

        # Get action from model or rule-based policy
        if self.ppo_model is not None:
            # Use trained model (deterministic for inference)
            action, _ = self.ppo_model.predict(observation, deterministic=True)
        else:
            # Use rule-based policy
            action = self._rule_based_policy(observation)

        # Convert action to steering
        steering = self.env.action_to_steering(action)

        return int(action), steering

    def _rule_based_policy(self, observation) -> int:
        """
        Simple rule-based policy for testing.

        Args:
            observation: Numpy array [ray_distances(5), ray_hits(5), speed(1)]

        Returns:
            Action: 0 (left), 1 (straight), or 2 (right)
        """
        import random

        # Parse observation
        ray_distances = observation[:5]
        ray_hits = observation[5:10]
        # speed = observation[10]

        # Forward ray is at index 0
        forward_dist = ray_distances[0]
        forward_hit = ray_hits[0]

        # Side rays: Right at index 3, Left at index 4
        right_dist = ray_distances[3]
        left_dist = ray_distances[4]
        right_hit = ray_hits[3]
        left_hit = ray_hits[4]

        # Decision logic
        if forward_hit and forward_dist < 2.0:  # Obstacle ahead
            # Turn away from the closest side obstacle
            if right_hit and right_dist < left_dist:
                return 0  # Turn left
            elif left_hit:
                return 2  # Turn right
            else:
                return random.choice([0, 2])  # Random turn (not straight)

        # Fine adjustments based on side rays
        elif right_hit and right_dist < 1.5:
            return 0  # Turn left
        elif left_hit and left_dist < 1.5:
            return 2  # Turn right
        else:
            # Mostly go straight, occasionally explore
            return random.choices([0, 1, 2], weights=[0.1, 0.8, 0.1])[0]
