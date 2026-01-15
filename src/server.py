from enum import Enum
import time
import sys
import os

# Add the src directory to the path to import our environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import AutoDrivingEnv
from ppo_controller import PPOController
from helpers import DebugHelper
from connection_manager import ConnectionManager

HOST = "127.0.0.1"
PORT = 65432


class GameState(Enum):
    Rewards = 0
    Respawns = 0
    ElapsedTime = 0
    CarSpeed = 0


def main():
    """Main server loop."""
    # Initialize Gym environment
    custom_ray_lengths = [50.0, 30.0, 30.0, 20.0, 20.0]

    env = AutoDrivingEnv(
        max_ray_distances=custom_ray_lengths, max_speed=2.5, steering_speed_penalty=0.5
    )

    print("=" * 60)
    print("Initialized Gym Environment:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Max ray distances: {env.max_ray_distances}")
    print(f"  Max speed: {env.max_speed}")
    print("=" * 60 + "\n")

    # Initialize PPO controller
    controller = PPOController(env)
    # controller.load_model("path/to/model.zip")  # Uncomment when model is ready

    # Initialize connection manager
    conn_manager = ConnectionManager(host=HOST, port=PORT)

    try:
        # Create server
        conn_manager.create_server()
        print(f"PPO Server listening on {HOST}:{PORT}")
        print("Waiting for Unity client connection...\n")

        episode_count = 0

        while True:  # Keep listening forever
            # Accept new connection
            try:
                if conn_manager.accept_client():
                    print(
                        f"Unity client connected from {conn_manager.client_address}\n"
                    )
                else:
                    continue
            except Exception as e:
                DebugHelper.log_error(f"Failed to accept connection: {e}")
                continue

            # Start new episode
            episode_count += 1
            step_count = 0
            print(f"Starting Episode {episode_count}")

            # Reset environment for new episode
            observation, info = env.reset()

            # Handle messages from Unity client
            while conn_manager.is_connected:
                # Receive message from Unity
                message = conn_manager.receive_json()

                if message is None:
                    # Connection closed or error
                    print(f"Episode {episode_count} ended after {step_count} steps\n")
                    break

                # Parse message
                timestamp = f"{time.strftime('%Y-%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}"
                message_type = message.get("message", "unknown")
                message_id = message.get("id", 0)
                game_state = message.get("gameState", {})

                print(f"\n--- Message {message_id} at {timestamp} ---")
                DebugHelper.print_game_state_summary(game_state)

                if message_type == "game_state":
                    # Get action from controller
                    action, steering = controller.get_action(game_state)

                    # Execute step in environment (for tracking/logging)
                    observation, reward, terminated, truncated, info = env.step(action)
                    step_count += 1

                    # Prepare response
                    response = {"steering": steering}

                    # Send action to Unity
                    steering_direction = "LEFT" if steering == -1 else "RIGHT"
                    print(
                        f"Sending action - Steering: {steering} ({steering_direction})"
                    )
                    print(f"  Reward: {reward:.2f}, Step: {step_count}")

                    success = conn_manager.send_json(response)

                    if not success:
                        DebugHelper.log_error("Failed to send action. Connection lost.")
                        break

                    # Check if episode should end
                    if terminated:
                        print(f"Episode {episode_count} terminated (crashed/respawned)")
                    elif truncated:
                        print(f"Episode {episode_count} truncated (max steps reached)")

                else:
                    # Unknown message type - send default action
                    DebugHelper.warn(f"Unknown message type: {message_type}")
                    conn_manager.send_json({"steering": 1})

            # Disconnect client after episode ends
            conn_manager.disconnect_client()

    except KeyboardInterrupt:
        print("\n\nPPO Server shutting down...")

    except Exception as e:
        DebugHelper.log_error(f"Unexpected error in main loop: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        conn_manager.close_server()
        env.close()
        print("Server stopped.")


if __name__ == "__main__":
    main()
