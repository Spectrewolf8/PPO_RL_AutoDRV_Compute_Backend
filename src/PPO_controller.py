from typing import Dict, Any, Tuple
from environment import AutoDrivingEnv


class PPOController:
    """
    Controls the PPO agent interaction with the environment.
    Generates actions based on observations.
    """

    def __init__(self, env: AutoDrivingEnv):
        """
        Initialize the PPO controller.

        Args:
            env: AutoDrivingEnv instance
        """
        self.env = env
        self.ppo_model = None  # Placeholder for trained model

    def load_model(self, model_path: str) -> None:
        """
        Load a trained PPO model.

        Args:
            model_path: Path to the saved model
        """
        # TODO: Implement model loading
        # from stable_baselines3 import PPO
        # self.ppo_model = PPO.load(model_path)
        print("Model loading not implemented yet. Using rule-based policy.")

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
            # Use trained model
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
            Action: 0 (left) or 1 (right)
        """
        import random

        # Parse observation
        ray_distances = observation[:5]
        ray_hits = observation[5:10]
        speed = observation[10]

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
                return 1  # Turn right
            else:
                return random.choice([0, 1])  # Random turn

        # Fine adjustments based on side rays
        elif right_hit and right_dist < 1.5:
            return 0  # Turn left
        elif left_hit and left_dist < 1.5:
            return 1  # Turn right
        else:
            # Random exploration
            return random.choice([0, 1])
