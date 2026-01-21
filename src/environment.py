import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List


class AutoDrivingEnv(gym.Env):
    """
    Custom Gym Environment for RL-based Autodriving Car Game

    Observation Space:
        - 5 ray distances (continuous values, each with independent max length)
        - 5 ray hit indicators (binary values)
        - 1 car speed (continuous value, max 2.5, reduces by 0.5 when steering)
        Total: 11 continuous values

    Action Space:
        - Discrete(3): 0 = turn left (-1), 1 = straight (0), 2 = turn right (1)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_ray_distances: Optional[List[float]] = None,
        max_speed: float = 2.5,
        steering_speed_penalty: float = 0.5,
    ):
        """
        Initialize the AutoDriving environment.

        Args:
            render_mode: Rendering mode (e.g., 'human')
            max_ray_distances: List of 5 max distances for each ray [Forward, Fwd-Left, Fwd-Right, Right, Left]
                              If None, defaults to [100.0, 100.0, 100.0, 100.0, 100.0]
            max_speed: Maximum speed in Unity (linear velocity)
            steering_speed_penalty: Speed reduction when steering (not used in Python, handled by Unity)
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_speed = max_speed
        self.steering_speed_penalty = steering_speed_penalty

        # Set individual ray max distances
        if max_ray_distances is None:
            self.max_ray_distances = [100.0] * 5
        else:
            if len(max_ray_distances) != 5:
                raise ValueError("max_ray_distances must contain exactly 5 values")
            self.max_ray_distances = list(max_ray_distances)

        # Define action space: Discrete actions (0 = left, 1 = straight, 2 = right)
        self.action_space = spaces.Discrete(3)

        # Define observation space
        # 5 ray distances (each with independent max) + 5 ray hits (0 or 1) + 1 speed (0 to max_speed)
        low_bounds = [0.0] * 5 + [0.0] * 5 + [0.0]
        high_bounds = self.max_ray_distances + [1.0] * 5 + [max_speed]

        self.observation_space = spaces.Box(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            shape=(11,),
            dtype=np.float32,
        )

        # Internal state
        self._current_state = None
        self._episode_step = 0
        self._max_episode_steps = 1000  # Can be configured

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Initialize with default state (all rays clear, no speed)
        self._current_state = {
            "rayDistances": self.max_ray_distances.copy(),
            "rayHits": [0] * 5,
            "carSpeed": 0.0,
            "rewards": 0.0,
            "respawns": 0,
            "elapsedTime": 0.0,
        }

        self._episode_step = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: 0 for left (-1), 1 for straight (0), 2 for right (1)

        Returns:
            observation: New observation after taking action
            reward: Reward received
            terminated: Whether episode ended due to terminal condition
            truncated: Whether episode ended due to time limit
            info: Additional information dictionary
        """
        # Convert discrete action to steering value
        steering = self.action_to_steering(action)

        self._episode_step += 1

        # NOTE: In actual integration, this is where you would:
        # 1. Send the steering action to Unity via WebSocket/API
        # 2. Wait for Unity to process the action and update game physics
        # 3. Receive the new game state from Unity (via update_state() method)
        # 4. Unity handles speed calculation based on steering and physics
        #
        # For now, this is a placeholder that keeps the current state.
        # When integrated with Unity:
        #   - Unity updates carSpeed based on steering_speed_penalty
        #   - Unity calculates and sends back updated rayDistances, rayHits
        #   - Unity computes rewards based on game events

        # Get observation from current state (updated by Unity via update_state())
        observation = self._get_observation()

        # Calculate reward (placeholder - will be updated from Unity)
        reward = self._calculate_reward()

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._episode_step >= self._max_episode_steps

        info = self._get_info()
        info["steering"] = steering

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Convert current state to observation array.

        Returns:
            Numpy array with [ray_distances (5), ray_hits (5), speed (1)]

        NOTE: This retrieves the observation from the current internal state,
        which should be updated by Unity via the update_state() method before
        calling this function.
        """
        # Ensure ray distances don't exceed their respective maximums
        ray_distances = []
        for i, dist in enumerate(self._current_state["rayDistances"]):
            clamped_dist = min(dist, self.max_ray_distances[i])
            ray_distances.append(clamped_dist)

        ray_distances = np.array(ray_distances, dtype=np.float32)
        ray_hits = np.array(self._current_state["rayHits"], dtype=np.float32)

        # Clamp speed to max_speed
        # NOTE: Speed value comes from Unity, which handles speed calculation
        # based on steering actions and physics simulation
        car_speed = min(self._current_state["carSpeed"], self.max_speed)
        car_speed = np.array([car_speed], dtype=np.float32)

        observation = np.concatenate([ray_distances, ray_hits, car_speed])

        return observation

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on current state.
        This is a placeholder implementation.

        Returns:
            Reward value
        """
        # In actual implementation, you might want to use the reward from Unity
        # or calculate it based on state

        reward = 0.0

        # Example reward structure (customize as needed):
        # 1. Reward for maintaining speed
        speed = self._current_state["carSpeed"]
        reward += speed * 0.01  # Small positive reward for moving

        # 2. Penalty for hitting obstacles
        ray_hits = self._current_state["rayHits"]
        if ray_hits[0] == 1:  # Forward ray hit
            ray_dist = self._current_state["rayDistances"][0]
            if ray_dist < 2.0:
                reward -= 1.0  # Penalty for being too close to obstacle

        # 3. Use reward from Unity if available
        if "rewards" in self._current_state:
            reward += self._current_state["rewards"]

        return reward

    def _is_terminated(self) -> bool:
        """
        Check if episode should terminate.

        Returns:
            True if episode is terminated
        """
        # Example termination conditions:
        # 1. If respawns indicate a crash
        if self._current_state.get("respawns", 0) > 0:
            return True

        # 2. If speed is zero for too long (stuck)
        # This would require tracking state history

        return False

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about current state.

        Returns:
            Info dictionary
        """
        return {
            "episode_step": self._episode_step,
            "car_speed": self._current_state["carSpeed"],
            "elapsed_time": self._current_state.get("elapsedTime", 0.0),
            "respawns": self._current_state.get("respawns", 0),
            "max_ray_distances": self.max_ray_distances,
            "max_speed": self.max_speed,
        }

    def update_state(self, game_state: Dict[str, Any]) -> None:
        """
        Update the environment's internal state with data from Unity.

        This method should be called after sending an action to Unity and
        receiving the updated game state. Unity is responsible for:
        - Updating ray distances and hit indicators based on raycasts
        - Calculating car speed based on steering and physics
        - Computing rewards based on game events'
        - Tracking respawns and elapsed time

        Args:
            game_state: Dictionary containing game state from Unity
                Expected keys: rayDistances, rayHits, carSpeed, rewards, respawns, elapsedTime
        """
        self._current_state.update(game_state)

    def action_to_steering(self, action: int) -> int:
        """
        Convert action index to steering value for Unity.

        Args:
            action: Action index (0, 1, or 2)

        Returns:
            Steering value: -1 for left, 0 for straight, 1 for right
        """
        if action == 0:
            return -1  # Turn left
        elif action == 1:
            return 0  # Go straight
        elif action == 2:
            return 1  # Turn right
        else:
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, or 2.")

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment (optional implementation).
        For Unity-based environment, rendering is handled by Unity.
        """
        if self.render_mode == "human":
            # Print current state to console
            print(f"\n=== Step {self._episode_step} ===")
            print(f"Speed: {self._current_state['carSpeed']:.2f} / {self.max_speed}")

            ray_names = ["Forward", "Fwd-Left", "Fwd-Right", "Right", "Left"]
            print("Rays:")
            for i, name in enumerate(ray_names):
                dist = self._current_state["rayDistances"][i]
                max_dist = self.max_ray_distances[i]
                hit = self._current_state["rayHits"][i]
                status = "HIT" if hit else "CLEAR"
                print(f"  {name}: {dist:.2f}/{max_dist:.2f} ({status})")

        return None

    def close(self) -> None:
        """
        Clean up resources.
        """
        pass


# Example usage and testing
if __name__ == "__main__":
    # Create environment with custom ray lengths
    # Ray order: [Forward, Fwd-Left, Fwd-Right, Right, Left]
    custom_ray_lengths = [7.0, 4.5, 4.5, 3.5, 3.5]

    env = AutoDrivingEnv(
        render_mode="human",
        max_ray_distances=custom_ray_lengths,
        max_speed=2.5,
        steering_speed_penalty=0.5,
    )

    # Test reset
    observation, info = env.reset()
    print("Initial observation shape:", observation.shape)
    print("Initial observation:", observation)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Info:", info)

    # Simulate some game states
    print("\n" + "=" * 50)
    print("Testing with simulated game states")
    print("=" * 50)

    # Simulate Unity sending game state
    simulated_state = {
        "rayDistances": [25.0, 15.0, 20.0, 10.0, 18.0],
        "rayHits": [0, 1, 0, 0, 1],
        "carSpeed": 2.5,
        "rewards": 1.0,
        "respawns": 0,
        "elapsedTime": 5.0,
    }

    env.update_state(simulated_state)

    # Test a few steps with all 3 actions
    print("\nTesting all action types:")
    for action in [0, 1, 2]:  # Test left, straight, right
        observation, reward, terminated, truncated, info = env.step(action)

        action_names = ["LEFT", "STRAIGHT", "RIGHT"]
        print(
            f"\nAction: {action} ({action_names[action]}) - Steering: {env.action_to_steering(action)}"
        )
        print(f"  Reward: {reward:.2f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        if terminated or truncated:
            break

    # Test a few random steps
    print("\n" + "=" * 50)
    print("Testing with random actions")
    print("=" * 50)

    for i in range(5):
        action = env.action_space.sample()  # Random action (0, 1, or 2)
        observation, reward, terminated, truncated, info = env.step(action)

        action_names = ["LEFT", "STRAIGHT", "RIGHT"]
        print(f"\nStep {i+1}:")
        print(
            f"  Action: {action} ({action_names[action]}) - Steering: {env.action_to_steering(action)}"
        )
        print(f"  Reward: {reward:.2f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        env.render()

        if terminated or truncated:
            break

    env.close()
    print("\n" + "=" * 50)
    print("Environment test complete!")
    print("=" * 50)
