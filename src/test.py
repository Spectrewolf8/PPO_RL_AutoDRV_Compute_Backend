# test.py
"""
Simulation Client for Testing PPO RL AutoDRV Server

This client simulates Unity game behavior to test server functionality:
- Connects to server via ZeroMQ REQ socket
- Receives and respects server tickrate configuration
- Simulates realistic game states (ray distances, speed, collisions, rewards)
- Runs multiple episodes with configurable steps
- Provides detailed logging and statistics
"""

import json
import logging
import time
import random
import zmq
from typing import Dict, Any, Optional, Tuple
from enum import Enum


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Different simulation behaviors for testing."""

    NORMAL = "normal"  # Realistic driving with occasional rewards/collisions
    AGGRESSIVE = "aggressive"  # More collisions and fast movements
    CAUTIOUS = "cautious"  # Slow, careful driving with fewer collisions
    REWARD_HUNTING = "reward_hunting"  # Frequent reward collection


class TestClient:
    """
    Simulation client that mimics Unity game behavior for testing the server.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 65432,
        num_episodes: int = 3,
        max_steps_per_episode: int = 100,
        simulation_mode: SimulationMode = SimulationMode.NORMAL,
    ):
        """
        Initialize the test client.

        Args:
            host: Server host address
            port: Server port number
            num_episodes: Number of episodes to simulate
            max_steps_per_episode: Maximum steps per episode
            simulation_mode: Simulation behavior mode
        """
        self.host = host
        self.port = port
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.simulation_mode = simulation_mode

        # ZeroMQ setup
        self.context = None
        self.socket = None
        self.endpoint = f"tcp://{host}:{port}"

        # Server configuration (received from server)
        self.server_tickrate = None
        self.tick_interval = None

        # Simulation state
        self.current_speed = 0.0
        self.max_speed = 2.5
        self.position = [0.0, 0.0]  # x, y position
        self.rotation = 0.0  # angle in degrees
        self.last_steering = 0  # -1, 0, 1

        # Episode tracking
        self.current_episode = 0
        self.episode_step = 0
        self.episode_rewards_collected = 0
        self.episode_collisions = 0

        # Statistics
        self.total_steps = 0
        self.total_rewards_collected = 0
        self.total_collisions = 0
        self.episode_statistics = []

        # Timing
        self.last_send_time = 0.0
        self.actual_send_rate = 0.0

        logger.info(f"TestClient initialized - Mode: {simulation_mode.value}")
        logger.info(f"Target: {host}:{port}")
        logger.info(f"Episodes: {num_episodes}, Max Steps: {max_steps_per_episode}")

    def connect(self) -> bool:
        """
        Connect to the server.

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            logger.info(f"Connecting to server at {self.endpoint}...")

            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)

            # Set socket options
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout

            self.socket.connect(self.endpoint)

            logger.info("Socket connected successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from the server."""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

        if self.context:
            try:
                self.context.term()
            except Exception:
                pass
            self.context = None

        logger.info("Disconnected from server")

    def _receive_server_configuration(self) -> bool:
        """
        Send initial message and receive server configuration.

        Returns:
            True if configuration received successfully
        """
        # Send initial connection message
        initial_message = self._generate_game_state(is_initial=True)

        logger.info("Sending initial message to server...")
        if not self._send_json(initial_message):
            return False

        # Receive configuration response
        config = self._receive_json()

        if config is None:
            logger.error("Failed to receive configuration from server")
            return False

        if config.get("type") == "config":
            self.server_tickrate = config.get("tickrate", 30)
            self.tick_interval = config.get("tick_interval_ms", 33.33) / 1000.0

            logger.info("=" * 60)
            logger.info("SERVER CONFIGURATION RECEIVED")
            logger.info(f"  Tickrate: {self.server_tickrate} Hz")
            logger.info(f"  Tick Interval: {self.tick_interval * 1000:.2f} ms")
            logger.info(
                f"  Max Episode Steps: {config.get('max_episode_steps', 'N/A')}"
            )
            logger.info("=" * 60)
            return True
        else:
            # Server might have sent action directly (old behavior)
            logger.warning("Server did not send configuration, using defaults")
            self.server_tickrate = 30
            self.tick_interval = 1.0 / 30
            return True

    def _send_json(self, data: Dict[str, Any]) -> bool:
        """Send JSON data to server."""
        try:
            json_bytes = json.dumps(data).encode()
            self.socket.send(json_bytes)
            return True
        except Exception as e:
            logger.error(f"Failed to send JSON: {e}")
            return False

    def _receive_json(self) -> Optional[Dict[str, Any]]:
        """Receive JSON data from server."""
        try:
            data = self.socket.recv()
            return json.loads(data.decode())
        except zmq.Again:
            logger.error("Receive timeout")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to receive JSON: {e}")
            return None

    def _generate_game_state(self, is_initial: bool = False) -> Dict[str, Any]:
        """
        Generate a simulated game state based on simulation mode.

        Args:
            is_initial: Whether this is the initial connection message

        Returns:
            Game state dictionary matching Unity format
        """
        # Simulate ray distances (5 rays: forward, forward-left, forward-right, left, right)
        ray_distances = self._simulate_ray_distances()
        ray_hits = [
            1 if d < max_d else 0
            for d, max_d in zip(ray_distances, [7.0, 4.5, 4.5, 3.5, 3.5])
        ]

        # Simulate speed changes based on steering
        if self.last_steering != 0:
            self.current_speed = max(0.0, min(self.max_speed, self.current_speed - 0.5))
        else:
            self.current_speed = min(self.max_speed, self.current_speed + 0.1)

        # Determine if reward collected or collision occurred
        reward_collected, collision_detected = self._simulate_events()

        # Build game state
        game_state = {
            "message": "initial" if is_initial else "gameState",
            "id": self.total_steps,
            "timestamp": time.time(),
            "gameState": {
                "rayDistances": ray_distances,
                "rayHits": ray_hits,
                "carSpeed": round(self.current_speed, 2),
                "rewardCollected": 1 if reward_collected else 0,
                "collisionDetected": 1 if collision_detected else 0,
                "respawns": 0,
                "position": {
                    "x": round(self.position[0], 2),
                    "y": round(self.position[1], 2),
                },
                "rotation": round(self.rotation, 2),
            },
        }

        return game_state

    def _simulate_ray_distances(self) -> list:
        """
        Simulate ray distances based on simulation mode.

        Returns:
            List of 5 ray distances
        """
        mode = self.simulation_mode

        if mode == SimulationMode.NORMAL:
            # Random but realistic distances
            return [
                round(random.uniform(2.0, 7.0), 2),  # forward
                round(random.uniform(1.5, 4.5), 2),  # forward-left
                round(random.uniform(1.5, 4.5), 2),  # forward-right
                round(random.uniform(1.0, 3.5), 2),  # left
                round(random.uniform(1.0, 3.5), 2),  # right
            ]

        elif mode == SimulationMode.AGGRESSIVE:
            # Closer obstacles, more danger
            return [
                round(random.uniform(0.5, 4.0), 2),
                round(random.uniform(0.5, 2.5), 2),
                round(random.uniform(0.5, 2.5), 2),
                round(random.uniform(0.3, 2.0), 2),
                round(random.uniform(0.3, 2.0), 2),
            ]

        elif mode == SimulationMode.CAUTIOUS:
            # More open space
            return [
                round(random.uniform(5.0, 7.0), 2),
                round(random.uniform(3.0, 4.5), 2),
                round(random.uniform(3.0, 4.5), 2),
                round(random.uniform(2.5, 3.5), 2),
                round(random.uniform(2.5, 3.5), 2),
            ]

        elif mode == SimulationMode.REWARD_HUNTING:
            # Open paths to simulate reward collection scenarios
            return [
                round(random.uniform(4.0, 7.0), 2),
                round(random.uniform(2.5, 4.5), 2),
                round(random.uniform(2.5, 4.5), 2),
                round(random.uniform(2.0, 3.5), 2),
                round(random.uniform(2.0, 3.5), 2),
            ]

        return [5.0, 3.0, 3.0, 2.5, 2.5]  # default

    def _simulate_events(self) -> Tuple[bool, bool]:
        """
        Simulate reward collection and collision events.

        Returns:
            Tuple of (reward_collected, collision_detected)
        """
        reward_collected = False
        collision_detected = False

        mode = self.simulation_mode

        if mode == SimulationMode.NORMAL:
            # 5% chance of reward, 3% chance of collision
            reward_collected = random.random() < 0.05
            collision_detected = random.random() < 0.03

        elif mode == SimulationMode.AGGRESSIVE:
            # 3% chance of reward, 10% chance of collision
            reward_collected = random.random() < 0.03
            collision_detected = random.random() < 0.10

        elif mode == SimulationMode.CAUTIOUS:
            # 7% chance of reward, 1% chance of collision
            reward_collected = random.random() < 0.07
            collision_detected = random.random() < 0.01

        elif mode == SimulationMode.REWARD_HUNTING:
            # 15% chance of reward, 2% chance of collision
            reward_collected = random.random() < 0.15
            collision_detected = random.random() < 0.02

        # Track events
        if reward_collected:
            self.episode_rewards_collected += 1
            self.total_rewards_collected += 1

        if collision_detected:
            self.episode_collisions += 1
            self.total_collisions += 1

        return reward_collected, collision_detected

    def _update_simulation_state(self, steering: float) -> None:
        """
        Update simulated position and rotation based on steering action.

        Args:
            steering: Steering value from server (-1, 0, 1)
        """
        self.last_steering = int(steering)

        # Update rotation based on steering
        rotation_speed = 45.0  # degrees per step
        self.rotation += steering * rotation_speed
        self.rotation = self.rotation % 360  # Keep in [0, 360]

        # Update position based on speed and rotation
        import math

        rad = math.radians(self.rotation)
        self.position[0] += self.current_speed * math.cos(rad) * 0.1
        self.position[1] += self.current_speed * math.sin(rad) * 0.1

    def _start_episode(self) -> None:
        """Initialize a new episode."""
        self.current_episode += 1
        self.episode_step = 0
        self.episode_rewards_collected = 0
        self.episode_collisions = 0

        # Reset simulation state
        self.current_speed = 1.0
        self.position = [0.0, 0.0]
        self.rotation = 0.0
        self.last_steering = 0

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"EPISODE {self.current_episode} STARTED")
        logger.info("=" * 80)

    def _end_episode(self, episode_reward: float, reason: str = "completed") -> None:
        """
        End the current episode and log statistics.

        Args:
            episode_reward: Total reward for the episode
            reason: Reason for episode ending
        """
        # Store episode statistics
        stats = {
            "episode": self.current_episode,
            "steps": self.episode_step,
            "total_reward": episode_reward,
            "avg_reward": episode_reward / max(self.episode_step, 1),
            "rewards_collected": self.episode_rewards_collected,
            "collisions": self.episode_collisions,
            "reason": reason,
        }
        self.episode_statistics.append(stats)

        # Log episode summary
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"EPISODE {self.current_episode} ENDED - {reason.upper()}")
        logger.info("-" * 80)
        logger.info(f"  Steps: {self.episode_step}")
        logger.info(f"  Total Reward: {episode_reward:.2f}")
        logger.info(f"  Average Reward: {stats['avg_reward']:.3f}")
        logger.info(f"  Rewards Collected: {self.episode_rewards_collected}")
        logger.info(f"  Collisions: {self.episode_collisions}")
        logger.info(f"  Avg Send Rate: {self.actual_send_rate:.1f} Hz")
        logger.info("=" * 80)
        logger.info("")

    def run(self) -> bool:
        """
        Run the simulation client.

        Returns:
            True if simulation completed successfully
        """
        try:
            # Connect to server
            if not self.connect():
                logger.error("Failed to connect to server")
                return False

            # Receive server configuration
            if not self._receive_server_configuration():
                logger.error("Failed to receive server configuration")
                return False

            logger.info("")
            logger.info("#" * 80)
            logger.info(f"# STARTING SIMULATION - {self.num_episodes} EPISODES")
            logger.info(f"# Mode: {self.simulation_mode.value.upper()}")
            logger.info(f"# Server Tickrate: {self.server_tickrate} Hz")
            logger.info("#" * 80)
            logger.info("")

            # Run episodes
            for episode_num in range(self.num_episodes):
                success = self._run_episode()

                if not success:
                    logger.error(f"Episode {episode_num + 1} failed")
                    break

                # Small delay between episodes
                if episode_num < self.num_episodes - 1:
                    logger.info("Waiting 2 seconds before next episode...")
                    time.sleep(2.0)

            # Print final statistics
            self._print_final_statistics()

            return True

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)
            return False
        finally:
            self.disconnect()

    def _run_episode(self) -> bool:
        """
        Run a single episode.

        Returns:
            True if episode completed successfully
        """
        self._start_episode()

        episode_reward = 0.0
        episode_terminated = False
        episode_truncated = False

        self.last_send_time = time.time()

        while self.episode_step < self.max_steps_per_episode:
            # Respect server tickrate
            current_time = time.time()
            time_since_last_send = current_time - self.last_send_time

            if time_since_last_send < self.tick_interval:
                # Wait to match server tickrate
                sleep_time = self.tick_interval - time_since_last_send
                time.sleep(sleep_time)
                current_time = time.time()

            # Calculate actual send rate
            self.actual_send_rate = (
                1.0 / (current_time - self.last_send_time)
                if self.last_send_time > 0
                else 0
            )
            self.last_send_time = current_time

            # Generate and send game state
            game_state = self._generate_game_state()

            if not self._send_json(game_state):
                logger.error("Failed to send game state")
                return False

            # Receive server response
            response = self._receive_json()

            if response is None:
                logger.error("Failed to receive server response")
                return False

            # Process server response
            steering = response.get("steering", 0)
            reward = response.get("reward", 0.0)
            episode_reward = response.get("episode_reward", 0.0)
            episode_terminated = response.get("terminated", False)
            episode_truncated = response.get("truncated", False)

            # Update simulation state
            self._update_simulation_state(steering)

            # Log progress periodically
            if self.episode_step % 10 == 0:
                logger.info(
                    f"  Step {self.episode_step:3d} | "
                    f"Steering: {steering:+.0f} | "
                    f"Speed: {self.current_speed:.2f} | "
                    f"Reward: {reward:+.1f} | "
                    f"Total: {episode_reward:+.1f} | "
                    f"Rate: {self.actual_send_rate:.1f} Hz"
                )

            self.episode_step += 1
            self.total_steps += 1

            # Check if episode should end
            if episode_terminated:
                self._end_episode(
                    episode_reward, reason="terminated (collision/respawn)"
                )
                return True

            if episode_truncated:
                self._end_episode(episode_reward, reason="truncated (max steps)")
                return True

        # Episode completed normally
        self._end_episode(episode_reward, reason="completed (client max steps)")
        return True

    def _print_final_statistics(self) -> None:
        """Print final simulation statistics."""
        logger.info("")
        logger.info("#" * 80)
        logger.info("# SIMULATION COMPLETE - FINAL STATISTICS")
        logger.info("#" * 80)
        logger.info("")
        logger.info(f"Total Episodes: {self.current_episode}")
        logger.info(f"Total Steps: {self.total_steps}")
        logger.info(f"Total Rewards Collected: {self.total_rewards_collected}")
        logger.info(f"Total Collisions: {self.total_collisions}")
        logger.info("")

        if self.episode_statistics:
            logger.info("Episode Breakdown:")
            logger.info("-" * 80)

            for stats in self.episode_statistics:
                logger.info(
                    f"  Episode {stats['episode']:2d}: "
                    f"{stats['steps']:3d} steps, "
                    f"Reward: {stats['total_reward']:+7.2f}, "
                    f"Collected: {stats['rewards_collected']}, "
                    f"Collisions: {stats['collisions']}, "
                    f"End: {stats['reason']}"
                )

            logger.info("-" * 80)

            # Calculate averages
            avg_steps = sum(s["steps"] for s in self.episode_statistics) / len(
                self.episode_statistics
            )
            avg_reward = sum(s["total_reward"] for s in self.episode_statistics) / len(
                self.episode_statistics
            )
            avg_collected = sum(
                s["rewards_collected"] for s in self.episode_statistics
            ) / len(self.episode_statistics)

            logger.info("")
            logger.info("Averages:")
            logger.info(f"  Steps per Episode: {avg_steps:.1f}")
            logger.info(f"  Reward per Episode: {avg_reward:+.2f}")
            logger.info(f"  Rewards Collected per Episode: {avg_collected:.1f}")

        logger.info("")
        logger.info("#" * 80)


def main():
    """Main entry point for the test client."""

    # Test configuration
    HOST = "127.0.0.1"
    PORT = 65432
    NUM_EPISODES = 3
    MAX_STEPS_PER_EPISODE = 50
    SIMULATION_MODE = SimulationMode.NORMAL

    logger.info("")
    logger.info("*" * 80)
    logger.info("* PPO RL AUTODRV - SIMULATION TEST CLIENT")
    logger.info("*" * 80)
    logger.info("")

    # Create and run test client
    client = TestClient(
        host=HOST,
        port=PORT,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        simulation_mode=SIMULATION_MODE,
    )

    success = client.run()

    if success:
        logger.info("")
        logger.info("✓ Simulation completed successfully")
        logger.info("")
    else:
        logger.error("")
        logger.error("✗ Simulation failed")
        logger.error("")


if __name__ == "__main__":
    main()
