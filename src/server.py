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


def main():
    """Main server loop."""
    # Initialize Gym environment with updated parameters
    custom_ray_lengths = [
        7.0,
        4.5,
        4.5,
        3.5,
        3.5,
    ]  # [Forward, Fwd-Left, Fwd-Right, Right, Left]

    env = AutoDrivingEnv(
        max_ray_distances=custom_ray_lengths,
        max_speed=2.5,
        steering_speed_penalty=0.5,
        reward_collected_value=10.0,
        collision_penalty=-10.0,
        survival_reward=0.1,
    )

    print("=" * 60)
    print("Initialized Gym Environment:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Max ray distances: {env.max_ray_distances}")
    print(f"  Max speed: {env.max_speed}")
    print("  Reward structure:")
    print(f"    - Survival reward: +{env.survival_reward} per step")
    print(f"    - Reward collected: +{env.reward_collected_value}")
    print(f"    - Collision penalty: {env.collision_penalty}")
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
            total_reward = 0.0
            rewards_collected = 0
            print(f"\n{'='*60}")
            print(f"Starting Episode {episode_count}")
            print(f"{'='*60}")

            # Reset environment for new episode
            observation, info = env.reset()

            # Handle messages from Unity client
            while conn_manager.is_connected:
                # Receive message from Unity
                message = conn_manager.receive_json()

                if message is None:
                    # Connection closed or error
                    print(f"\n{'='*60}")
                    print(f"Episode {episode_count} Summary:")
                    print(f"  Total steps: {step_count}")
                    print(f"  Total reward: {total_reward:.2f}")
                    print(f"  Rewards collected: {rewards_collected}")
                    print(f"{'='*60}\n")
                    break

                # Parse message
                timestamp = f"{time.strftime('%Y-%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}"
                message_type = message.get("message", "unknown")
                message_id = message.get("id", 0)
                game_state = message.get("gameState", {})

                print(f"\n--- Message {message_id} at {timestamp} ---")

                # Validate game state has all required fields
                required_fields = [
                    "rayDistances",
                    "rayHits",
                    "carSpeed",
                    "rewardCollected",
                    "collisionDetected",
                    "respawns",
                    "elapsedTime",
                ]
                missing_fields = [
                    field for field in required_fields if field not in game_state
                ]

                if missing_fields:
                    DebugHelper.warn(f"Missing fields in game state: {missing_fields}")
                    # Provide defaults for missing fields
                    if "rewardCollected" not in game_state:
                        game_state["rewardCollected"] = 0
                    if "collisionDetected" not in game_state:
                        game_state["collisionDetected"] = 0

                # Print detailed game state
                DebugHelper.print_game_state_summary(game_state)

                if message_type == "game_state":
                    # Update environment state with Unity data
                    env.update_state(game_state)

                    # Get action from controller
                    action, steering = controller.get_action(game_state)

                    # Execute step in environment (calculates reward, checks termination)
                    observation, reward, terminated, truncated, info = env.step(action)
                    step_count += 1
                    total_reward += reward

                    # Track rewards collected
                    if game_state.get("rewardCollected", 0) == 1:
                        rewards_collected += 1

                    # Prepare response (only send steering to Unity)
                    response = {"steering": steering}

                    # Log action and results
                    action_names = {-1: "LEFT", 0: "STRAIGHT", 1: "RIGHT"}
                    print(f"Action: Steering {steering} ({action_names[steering]})")
                    print(f"  Step: {step_count}")
                    print(f"  Reward this step: {reward:.2f}")
                    print(f"  Total reward: {total_reward:.2f}")

                    # Log special events
                    if game_state.get("rewardCollected", 0) == 1:
                        print("REWARD COLLECTED!")
                    if game_state.get("collisionDetected", 0) == 1:
                        print("COLLISION DETECTED!")

                    # Send action to Unity
                    success = conn_manager.send_json(response)

                    if not success:
                        DebugHelper.log_error("Failed to send action. Connection lost.")
                        break

                    # Check if episode should end
                    if terminated:
                        print(f"\nEpisode {episode_count} TERMINATED (collision/crash)")
                        # Send final response before disconnecting
                        time.sleep(0.1)  # Give Unity time to process
                        break
                    elif truncated:
                        print(
                            f"\nEpisode {episode_count} TRUNCATED (max steps: {env._max_episode_steps})"
                        )
                        break

                elif message_type == "reset":
                    # Unity requests episode reset
                    print("Unity requested reset")
                    observation, info = env.reset()
                    conn_manager.send_json({"status": "reset_complete"})

                else:
                    # Unknown message type - send default action (go straight)
                    DebugHelper.warn(f"Unknown message type: {message_type}")
                    conn_manager.send_json({"steering": 0})

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
