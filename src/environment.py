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

    Reward Structure:
        - Python calculates rewards based on Unity signals
        - Unity sends: rewardCollected (0/1), collisionDetected (0/1)
        - Server-side reward calculation provides flexibility for tuning
        - Additional reward given for driving straight to encourage stable driving
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_ray_distances: Optional[List[float]] = None,
        max_speed: float = 2.5,
        steering_speed_penalty: float = 0.5,
        reward_collected_value: float = 10.0,
        collision_penalty: float = -10.0,
        survival_reward: float = 0.1,
        straight_driving_reward: float = 0.05,
    ):
        """
        Initialize the AutoDriving environment.

        Args:
            render_mode: Rendering mode (e.g., 'human')
            max_ray_distances: List of 5 max distances for each ray [Forward, Fwd-Left, Fwd-Right, Right, Left]
                              If None, defaults to [100.0, 100.0, 100.0, 100.0, 100.0]
            max_speed: Maximum speed in Unity (linear velocity)
            steering_speed_penalty: Speed reduction when steering (not used in Python, handled by Unity)
            reward_collected_value: Reward value when collecting a reward object
            collision_penalty: Penalty value when collision is detected
            survival_reward: Small reward per step for staying alive (encourages survival without biasing steering)
            straight_driving_reward: Reward for driving straight (encourages stable, straight-line driving)
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_speed = max_speed
        self.steering_speed_penalty = steering_speed_penalty

        # Reward configuration (server-side calculation)
        self.reward_collected_value = reward_collected_value
        self.collision_penalty = collision_penalty
        self.survival_reward = survival_reward
        self.straight_driving_reward = straight_driving_reward

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
        self._last_action = None  # Track last action for reward calculation

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
            "rewardCollected": 0,
            "collisionDetected": 0,
            "respawns": 0,
            "elapsedTime": 0.0,
        }

        self._episode_step = 0
        self._last_action = None  # Reset last action

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

        # Store action for reward calculation
        self._last_action = action

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
        #   - Unity sends rewardCollected (0/1) and collisionDetected (0/1) signals
        #   - Python server calculates actual reward values

        # Get observation from current state (updated by Unity via update_state())
        observation = self._get_observation()

        # Calculate reward based on Unity signals
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
        Calculate reward based on Unity signals.

        Unity sends:
            - rewardCollected: 0 or 1 (1 when reward object is collected)
            - collisionDetected: 0 or 1 (1 when collision occurs)

        Python calculates the actual reward values based on these signals.
        This allows for flexible reward tuning without modifying Unity code.

        Reward structure:
            - Survival: Small positive reward per step (encourages staying alive)
            - Straight driving: Small positive reward for driving straight (encourages stable driving)
            - Reward collection: Large positive reward
            - Collision: Large negative penalty

        Returns:
            Reward value
        """
        reward = 0.0

        # 1. Survival reward - given every step the car stays alive
        # This encourages the model to avoid collisions without biasing toward/against steering
        reward += self.survival_reward

        # 2. Straight driving reward - given when driving straight (action 1)
        # This encourages stable, straight-line driving when appropriate
        if self._last_action == 1:
            reward += self.straight_driving_reward

        # 3. Reward for collecting reward objects (sent from Unity)
        if self._current_state.get("rewardCollected", 0) == 1:
            reward += self.reward_collected_value

        # 4. Penalty for collisions (sent from Unity)
        if self._current_state.get("collisionDetected", 0) == 1:
            reward += self.collision_penalty

        return reward

    def _is_terminated(self) -> bool:
        """
        Check if episode should terminate.

        Returns:
            True if episode is terminated
        """
        # Episode terminates on collision
        if self._current_state.get("collisionDetected", 0) == 1:
            return True

        # Or if respawns indicate a crash
        if self._current_state.get("respawns", 0) > 0:
            return True

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
            "reward_collected": self._current_state.get("rewardCollected", 0),
            "collision_detected": self._current_state.get("collisionDetected", 0),
            "max_ray_distances": self.max_ray_distances,
            "max_speed": self.max_speed,
        }

    def update_state(self, game_state: Dict[str, Any]) -> None:
        """
        Update the environment's internal state with data from Unity.

        Unity is responsible for:
        - Updating ray distances and hit indicators based on raycasts
        - Calculating car speed based on steering and physics
        - Detecting reward collection (sending rewardCollected signal: 0 or 1)
        - Detecting collisions (sending collisionDetected signal: 0 or 1)
        - Tracking respawns and elapsed time

        Python server calculates actual reward values based on these signals.

        Args:
            game_state: Dictionary containing game state from Unity
                Expected keys:
                    - rayDistances: List[float] (5 values)
                    - rayHits: List[int] (5 values, 0 or 1)
                    - carSpeed: float
                    - rewardCollected: int (0 or 1)
                    - collisionDetected: int (0 or 1)
                    - respawns: int
                    - elapsedTime: float
        """
        self._current_state.update(game_state)

    def action_to_steering(self, action: int) -> int:
        """
        Convert action index to steering value for Unity.

        This translation layer converts the RL model's discrete action space
        to the game engine's input format.

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

            # Display signals from Unity
            if self._current_state.get("rewardCollected", 0) == 1:
                print("REWARD COLLECTED!")
            if self._current_state.get("collisionDetected", 0) == 1:
                print("COLLISION DETECTED!")

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
        reward_collected_value=10.0,
        collision_penalty=-10.0,
        survival_reward=0.1,
        straight_driving_reward=0.05,
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

    # Simulate Unity sending game state (normal movement)
    simulated_state = {
        "rayDistances": [25.0, 15.0, 20.0, 10.0, 18.0],
        "rayHits": [0, 1, 0, 0, 1],
        "carSpeed": 2.5,
        "rewardCollected": 0,
        "collisionDetected": 0,
        "respawns": 0,
        "elapsedTime": 5.0,
    }

    env.update_state(simulated_state)

    # Test a few steps with all 3 actions
    print("\nTesting all action types:")
    for action in [0, 1, 2]:  # Test left, straight, right
        observation, reward, terminated, truncated, info = env.step(action)

        action_names = ["LEFT", "STRAIGHT", "RIGHT"]
        print(f"\nAction: {action} ({action_names[action]}) - Steering: {env.action_to_steering(action)}")
        reward_desc = "survival reward"
        if action == 1:
            reward_desc += " + straight driving reward"
        print(f"  Reward: {reward:.2f} ({reward_desc})")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        if terminated or truncated:
            break

    # Test reward collection
    print("\n" + "=" * 50)
    print("Testing reward collection")
    print("=" * 50)

    simulated_state["rewardCollected"] = 1
    env.update_state(simulated_state)
    observation, reward, terminated, truncated, info = env.step(1)
    print(f"Reward collected! Total reward: {reward:.2f} (survival + collection)")
    env.render()

    # Reset signal for next step
    simulated_state["rewardCollected"] = 0
    env.update_state(simulated_state)

    # Test collision
    print("\n" + "=" * 50)
    print("Testing collision detection")
    print("=" * 50)

    simulated_state["collisionDetected"] = 1
    env.update_state(simulated_state)
    observation, reward, terminated, truncated, info = env.step(1)
    print(f"Collision detected! Total reward: {reward:.2f} (survival + collision penalty), Terminated: {terminated}")
    env.render()

    env.close()
    print("\n" + "=" * 50)
    print("Environment test complete!")
    print("=" * 50)
