# server.py
import json
import logging
import time
from typing import Dict, Any, Optional
from enum import Enum
from connection_manager import ConnectionManager
from environment import AutoDrivingEnv
from ppo_controller import PPOController


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ServerState(Enum):
    """Server operational states."""

    IDLE = 0
    WAITING_FOR_CLIENT = 1
    EPISODE_RUNNING = 2
    EPISODE_ENDED = 3
    SHUTTING_DOWN = 4


class GameServer:
    """
    Main server for handling Unity game communication via ZeroMQ.

    The server:
    - Receives game state from Unity client
    - Processes state through PPO agent
    - Sends steering commands back to Unity
    - Handles episode lifecycle (rewards, collisions, resets)
    - Uses configurable tickrate to control update frequency
    - Synchronizes tickrate with client on connection
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 65432,
        tickrate: int = 30,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the game server.

        Args:
            host: Server host address
            port: Server port number
            tickrate: Updates per second (how many game state updates to process per second)
            model_path: Path to trained PPO model (optional, uses rule-based if None)
        """
        self.host = host
        self.port = port
        self.tickrate = tickrate
        self.tick_interval = 1.0 / tickrate  # Time between ticks in seconds
        self.model_path = model_path

        # Components
        # Set receive timeout to 3x the tick interval (allows for some delay)
        # Minimum 2 seconds to avoid false disconnects
        receive_timeout_ms = max(2000, int(self.tick_interval * 3000))

        self.connection_manager = ConnectionManager(
            host=host, port=port, timeout_ms=1000, receive_timeout_ms=receive_timeout_ms
        )

        # Initialize environment with ray distances from spec
        self.env = AutoDrivingEnv(
            max_ray_distances=[
                7.0,
                4.5,
                4.5,
                3.5,
                3.5,
            ],  # front, front-left, front-right, left, right
            max_speed=2.5,
            steering_speed_penalty=0.5,
            reward_collected_value=15.0,
            collision_penalty=-10.0,
            survival_reward=0.1,
        )

        self.controller = PPOController(self.env)

        # Server state
        self.state = ServerState.IDLE
        self.running = False

        # Episode tracking
        self.current_episode = 0
        self.episode_step = 0
        self.episode_reward = 0.0
        self.total_episodes = 0
        self.total_steps = 0

        # Connection tracking
        self.first_message_received = False

        # Performance metrics
        self.last_tick_time = 0.0
        self.actual_tickrate = 0.0
        self.message_count = 0

        logger.info(
            f"GameServer initialized - Host: {host}, Port: {port}, Tickrate: {tickrate} Hz"
        )

    def load_model(self, model_path: str) -> None:
        """
        Load a trained PPO model.

        Args:
            model_path: Path to the saved model
        """
        logger.info(f"Loading PPO model from {model_path}")
        self.controller.load_model(model_path)

    def start(self) -> None:
        """Start the server and begin listening for connections."""
        logger.info("Starting game server...")

        try:
            # Create and bind server socket
            self.connection_manager.create_server()
            logger.info(f"Server listening on {self.host}:{self.port}")

            self.running = True
            self.state = ServerState.WAITING_FOR_CLIENT

            # Load model if path provided
            if self.model_path:
                self.load_model(self.model_path)

            # Main server loop
            self._run_server_loop()

        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            self.shutdown()

    def _run_server_loop(self) -> None:
        """Main server loop handling client connections and episodes."""
        while self.running:
            if self.state == ServerState.WAITING_FOR_CLIENT:
                self._wait_for_client()

            elif self.state == ServerState.EPISODE_RUNNING:
                self._run_episode()

            elif self.state == ServerState.EPISODE_ENDED:
                self._handle_episode_end()

            elif self.state == ServerState.SHUTTING_DOWN:
                break

            # Small sleep to prevent busy waiting
            time.sleep(0.001)

    def _wait_for_client(self) -> None:
        """Wait for Unity client to connect."""
        logger.info("Waiting for Unity client to connect...")

        while self.running and self.state == ServerState.WAITING_FOR_CLIENT:
            # Try to accept client (non-blocking with timeout)
            if self.connection_manager.accept_client():
                logger.info("Unity client connected!")
                self._start_new_episode()
                self.state = ServerState.EPISODE_RUNNING
                break

            time.sleep(0.1)  # Check every 100ms

    def _start_new_episode(self) -> None:
        """Start a new episode."""
        self.current_episode += 1
        self.episode_step = 0
        self.episode_reward = 0.0
        self.message_count = 0
        self.last_tick_time = time.time()
        # DON'T reset first_message_received here - it's per connection, not per episode

        # Reset environment
        observation, info = self.env.reset()

        logger.info(f"=== Episode {self.current_episode} Started ===")
        logger.info(f"Initial observation shape: {observation.shape}")

    def _run_episode(self) -> None:
        """Run episode loop, processing game states and sending actions."""
        try:
            # Check if client is still connected before receiving
            if not self.connection_manager.is_connected:
                logger.warning("Client connection lost")
                self._handle_client_disconnect()
                return

            # Receive game state from Unity
            game_state = self.connection_manager.receive_json()

            if game_state is None:
                # Connection lost or timeout
                logger.warning(
                    "No data received from client (connection lost or timeout)"
                )
                self._handle_client_disconnect()
                return

            # Handle first message - send configuration including tickrate
            if not self.first_message_received:
                self.first_message_received = True
                self._send_initial_configuration()
                return

            # Enforce tickrate
            current_time = time.time()
            time_since_last_tick = current_time - self.last_tick_time

            if time_since_last_tick < self.tick_interval:
                # Too fast, wait before processing
                sleep_time = self.tick_interval - time_since_last_tick
                time.sleep(sleep_time)
                current_time = time.time()

            # Calculate actual tickrate
            self.actual_tickrate = (
                1.0 / (current_time - self.last_tick_time)
                if self.last_tick_time > 0
                else 0
            )
            self.last_tick_time = current_time

            # Process the game state
            response = self._process_game_state(game_state)

            # Send response back to Unity
            if not self.connection_manager.send_json(response):
                logger.error("Failed to send response to Unity client")
                self._handle_client_disconnect()
                return

            self.message_count += 1

            # Check if episode should end
            if self._should_episode_end(game_state):
                self.state = ServerState.EPISODE_ENDED

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received from client: {e}")
        except Exception as e:
            logger.error(f"Error during episode: {e}", exc_info=True)
            self._handle_client_disconnect()

    def _send_initial_configuration(self) -> None:
        """
        Send initial configuration to Unity client including tickrate.
        This is sent as response to the first message from client.
        """
        config = {
            "type": "config",
            "tickrate": self.tickrate,
            "tick_interval_ms": self.tick_interval * 1000,
            "max_episode_steps": self.env._max_episode_steps,
            "message": "Server configuration. Please synchronize your update rate.",
        }

        if self.connection_manager.send_json(config):
            logger.info(
                f"Sent configuration to Unity client: Tickrate={self.tickrate} Hz, Interval={self.tick_interval*1000:.2f}ms"
            )
        else:
            logger.error("Failed to send configuration to Unity client")
            self._handle_client_disconnect()

    def _process_game_state(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process received game state and generate action.

        Args:
            game_state: Game state dictionary from Unity

        Returns:
            Response dictionary with steering command and episode statistics
        """
        # Extract message info
        message_type = game_state.get("message", "unknown")
        message_id = game_state.get("id", -1)
        state_data = game_state.get("gameState", {})

        if self.episode_step:  # Log every 10 steps to reduce spam
            logger.info(
                f"Step {self.episode_step} | MsgID: {message_id} | "
                f"Speed: {state_data.get('carSpeed', 0):.2f} | "
                f"Rays: {[f'{d:.1f}' for d in state_data.get('rayDistances', [])]}"
            )

        # Handle reset message
        if message_type == "reset":
            logger.info("Received reset request from Unity")
            self.state = ServerState.EPISODE_ENDED
            return {"steering": 0}

        # Update environment with new state
        self.env.update_state(state_data)

        # Get observation
        # observation = self.env._get_observation()

        # Calculate reward
        reward = self._calculate_reward(state_data)
        self.episode_reward += reward

        # Check for terminal conditions
        # terminated = self._is_terminated(state_data)
        # truncated = self.episode_step >= self.env._max_episode_steps

        # Get action from PPO controller
        action, steering = self.controller.get_action(state_data)

        # Log important events
        if state_data.get("rewardCollected", 0) == 1:
            logger.info(
                f"      Reward cube collected! Reward: +{self.env.reward_collected_value}"
            )

        if state_data.get("collisionDetected", 0) == 1:
            logger.warning(
                f"    Collision detected! Penalty: {self.env.collision_penalty}"
            )

        # Update step counter
        self.episode_step += 1
        self.total_steps += 1

        # Check termination status for Unity
        terminated = self._is_terminated(state_data)
        truncated = self.episode_step >= self.env._max_episode_steps

        # Prepare response with full feedback for Unity UI
        response = {
            "steering": steering,
            "reward": reward,
            "episode_reward": self.episode_reward,
            "step": self.episode_step,
            "total_steps": self.total_steps,
            "episode": self.current_episode,
            "total_episodes": self.total_episodes,
            "terminated": terminated,
            "truncated": truncated,
        }

        return response

    def _calculate_reward(self, state_data: Dict[str, Any]) -> float:
        """
        Calculate reward based on game state signals.

        Args:
            state_data: Game state data from Unity

        Returns:
            Calculated reward value
        """
        reward = 0.0

        # Survival reward (always given per step)
        reward += self.env.survival_reward

        # Reward for collecting reward cube
        if state_data.get("rewardCollected", 0) == 1:
            reward += self.env.reward_collected_value

        # Penalty for collision
        if state_data.get("collisionDetected", 0) == 1:
            reward += self.env.collision_penalty

        return reward

    def _is_terminated(self, state_data: Dict[str, Any]) -> bool:
        """
        Check if episode should terminate.

        Args:
            state_data: Game state data from Unity

        Returns:
            True if episode should terminate
        """
        # Collision causes termination
        if state_data.get("collisionDetected", 0) == 1:
            return True

        # Respawn causes termination
        if state_data.get("respawns", 0) > 0:
            return True

        return False

    def _should_episode_end(self, game_state: Dict[str, Any]) -> bool:
        """
        Determine if episode should end based on game state.

        Args:
            game_state: Full game state message from Unity

        Returns:
            True if episode should end
        """
        state_data = game_state.get("gameState", {})

        # Check termination conditions
        if self._is_terminated(state_data):
            return True

        # Check truncation (max steps)
        if self.episode_step >= self.env._max_episode_steps:
            logger.info(
                f"Episode truncated at max steps ({self.env._max_episode_steps})"
            )
            return True

        return False

    def _handle_episode_end(self) -> None:
        """Handle episode end, log statistics, and prepare for next episode."""
        self.total_episodes += 1

        # Log episode summary
        logger.info(f"=== Episode {self.current_episode} Ended ===")
        logger.info(f"Total Steps: {self.episode_step}")
        logger.info(f"Total Reward: {self.episode_reward:.2f}")
        logger.info(
            f"Average Reward: {self.episode_reward / max(self.episode_step, 1):.3f}"
        )
        logger.info(f"Messages Processed: {self.message_count}")
        logger.info(f"Average Tickrate: {self.actual_tickrate:.1f} Hz")
        logger.info("=" * 50)

        # Check if client is still connected
        if self.connection_manager.check_connection():
            # Client still connected, start new episode
            logger.info("Client still connected. Starting new episode...")
            self._start_new_episode()
            self.state = ServerState.EPISODE_RUNNING
        else:
            # Client disconnected
            logger.info("Client disconnected between episodes")
            self._handle_client_disconnect()

    def _handle_client_disconnect(self) -> None:
        """Handle Unity client disconnection and reset to initial state."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("CLIENT DISCONNECTED")
        logger.info("=" * 60)

        # Log session statistics if we had any episodes
        if self.total_episodes > 0:
            logger.info("Session Statistics:")
            logger.info(f"  Total Episodes Completed: {self.total_episodes}")
            logger.info(f"  Total Steps Processed: {self.total_steps}")
            logger.info(
                f"  Average Steps per Episode: {self.total_steps / max(self.total_episodes, 1):.1f}"
            )

        # Disconnect client at connection manager level
        self.connection_manager.disconnect_client()

        # Reset to initial state
        logger.info("Resetting server to initial state...")
        self.state = ServerState.WAITING_FOR_CLIENT
        self.first_message_received = False

        logger.info("Server ready for new client connection")
        logger.info("=" * 60)
        logger.info("")

    def set_tickrate(self, tickrate: int) -> None:
        """
        Change the server tickrate dynamically.

        Args:
            tickrate: New tickrate in Hz (updates per second)
        """
        if tickrate <= 0:
            logger.warning(f"Invalid tickrate {tickrate}, must be > 0")
            return

        self.tickrate = tickrate
        self.tick_interval = 1.0 / tickrate
        logger.info(
            f"Tickrate changed to {tickrate} Hz (interval: {self.tick_interval*1000:.2f}ms)"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get server statistics.

        Returns:
            Dictionary containing server statistics
        """
        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "current_episode": self.current_episode,
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
            "tickrate": self.tickrate,
            "actual_tickrate": self.actual_tickrate,
            "state": self.state.name,
            "connection_state": self.connection_manager.state.name,
        }

    def shutdown(self) -> None:
        """Shutdown the server gracefully."""
        logger.info("Shutting down server...")
        self.running = False
        self.state = ServerState.SHUTTING_DOWN

        # Disconnect client if connected
        if self.connection_manager.is_connected:
            self.connection_manager.disconnect_client()

        # Close server
        self.connection_manager.close_server()

        # Log final statistics
        stats = self.get_statistics()
        logger.info("Final Statistics:")
        logger.info(f"  Total Episodes: {stats['total_episodes']}")
        logger.info(f"  Total Steps: {stats['total_steps']}")
        logger.info(
            f"  Average Steps per Episode: {stats['total_steps'] / max(stats['total_episodes'], 1):.1f}"
        )

        logger.info("Server shutdown complete")


def main():
    """Main entry point for the server."""
    # Server configuration
    HOST = "127.0.0.1"
    PORT = 65432
    TICKRATE = 2  # 2 updates per second
    MODEL_PATH = None  # Set to model path if you have a trained model

    # Create and start server
    server = GameServer(
        host=HOST,
        port=PORT,
        tickrate=TICKRATE,
        model_path=MODEL_PATH,
    )

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        server.shutdown()


if __name__ == "__main__":
    main()
