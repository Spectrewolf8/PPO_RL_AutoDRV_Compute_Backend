import json
import socket
import time
import sys
import os

# Add the src directory to the path to import our environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import AutoDrivingEnv

HOST = "127.0.0.1"  # Listen on all available interfaces
PORT = 65432  # Port to listen on (matches Unity client)


def generate_ppo_action(env, game_state):
    """
    Generate a PPO action based on game state using the Gym environment.
    This is a placeholder - replace with actual PPO model inference.
    Returns discrete steering: -1 (left) or 1 (right)

    Args:
        env: AutoDrivingEnv instance
        game_state: Dictionary containing game state from Unity

    Returns:
        Dictionary with steering action
    """
    # Update environment with game state from Unity
    env.update_state(game_state)

    # Get observation from environment
    observation = env._get_observation()

    # TODO: Replace with actual PPO model prediction
    # action = ppo_model.predict(observation, deterministic=True)

    # For now, use rule-based logic
    action = rule_based_policy(observation)

    # Convert action to steering value
    steering = env.action_to_steering(action)

    return {"steering": steering, "action": int(action)}


def rule_based_policy(observation):
    """
    Simple rule-based policy for testing.
    Replace this with your trained PPO model.

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


def print_game_state_summary(game_state):
    """Print a summary of the received game state"""
    rewards = game_state.get("rewards", 0)
    respawns = game_state.get("respawns", 0)
    elapsed_time = game_state.get("elapsedTime", 0)
    car_speed = game_state.get("carSpeed", 0)

    # Ray information
    ray_distances = game_state.get("rayDistances", [])
    ray_hits = game_state.get("rayHits", [])

    print(
        f"Game State - Rewards: {rewards}, Respawns: {respawns}, Time: {elapsed_time:.1f}s, Speed: {car_speed:.2f}"
    )

    if len(ray_distances) >= 5:
        ray_names = ["Forward", "Fwd-Left", "Fwd-Right", "Right", "Left"]
        for i, (name, dist, hit) in enumerate(zip(ray_names, ray_distances, ray_hits)):
            status = "HIT" if hit else "CLEAR"
            print(f"  {name}: {dist:.2f} ({status})")


def main():
    """Main server loop"""
    # Initialize Gym environment
    # Customize ray lengths based on your Unity setup
    custom_ray_lengths = [
        50.0,
        30.0,
        30.0,
        20.0,
        20.0,
    ]  # Forward, Fwd-L, Fwd-R, Right, Left

    env = AutoDrivingEnv(
        max_ray_distances=custom_ray_lengths, max_speed=2.5, steering_speed_penalty=0.5
    )

    print("Initialized Gym Environment:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Max ray distances: {env.max_ray_distances}")
    print(f"  Max speed: {env.max_speed}\n")

    # Reset environment for first episode
    observation, info = env.reset()
    episode_count = 0
    step_count = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow port reuse
        s.bind((HOST, PORT))
        s.listen()
        print(f"PPO Server listening on {HOST}:{PORT}")
        print("Waiting for Unity client connection...")

        while True:  # Keep listening forever
            try:
                conn, addr = s.accept()
                with conn:
                    print(f"\nUnity client connected from {addr}")
                    episode_count += 1
                    print(f"Starting Episode {episode_count}")

                    # Reset environment for new episode
                    observation, info = env.reset()
                    step_count = 0

                    while True:
                        try:
                            data = conn.recv(
                                4096
                            )  # Increased buffer for game state data
                            if not data:
                                print(f"Unity client {addr} disconnected normally")
                                print(
                                    f"Episode {episode_count} ended after {step_count} steps"
                                )
                                break

                            timestamp = f"{time.strftime('%Y-%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}"

                            try:
                                # Parse the incoming message
                                message_data = json.loads(data.decode())
                                message_type = message_data.get("message", "unknown")
                                message_id = message_data.get("id", 0)
                                game_state = message_data.get("gameState", {})

                                print(f"\n--- Message {message_id} at {timestamp} ---")
                                print_game_state_summary(game_state)

                                if message_type == "game_state":
                                    # Generate PPO action using Gym environment
                                    action_result = generate_ppo_action(env, game_state)

                                    # Execute step in environment (for tracking/logging)
                                    observation, reward, terminated, truncated, info = (
                                        env.step(action_result["action"])
                                    )
                                    step_count += 1

                                    # Send action response
                                    response_data = {
                                        "steering": action_result["steering"]
                                    }

                                    response = json.dumps(response_data).encode()
                                    steering_direction = (
                                        "LEFT"
                                        if action_result["steering"] == -1
                                        else "RIGHT"
                                    )
                                    print(
                                        f"Sending action - Steering: {action_result['steering']} ({steering_direction})"
                                    )
                                    print(f"  Reward: {reward:.2f}, Step: {step_count}")

                                    # Check if episode should end
                                    if terminated:
                                        print(
                                            f"Episode {episode_count} terminated (crashed/respawned)"
                                        )
                                    elif truncated:
                                        print(
                                            f"Episode {episode_count} truncated (max steps reached)"
                                        )

                                    conn.sendall(response)

                                else:
                                    # Unknown message type
                                    response_data = {"steering": 1}  # Default to right
                                    response = json.dumps(response_data).encode()
                                    conn.sendall(response)

                            except json.JSONDecodeError as e:
                                print(f"JSON decode error: {e}")
                                # Send default action on JSON error
                                response_data = {"steering": 1}  # Default to right
                                response = json.dumps(response_data).encode()
                                conn.sendall(response)

                        except ConnectionResetError:
                            print(
                                f"Unity client {addr} disconnected unexpectedly (connection reset)"
                            )
                            print(
                                f"Episode {episode_count} ended after {step_count} steps"
                            )
                            break
                        except BrokenPipeError:
                            print(f"Unity client {addr} disconnected (broken pipe)")
                            print(
                                f"Episode {episode_count} ended after {step_count} steps"
                            )
                            break
                        except socket.error as e:
                            if e.errno == 104:  # Connection reset by peer
                                print(f"Unity client {addr} reset the connection")
                            else:
                                print(f"Socket error with Unity client {addr}: {e}")
                            print(
                                f"Episode {episode_count} ended after {step_count} steps"
                            )
                            break

            except KeyboardInterrupt:
                print("\nPPO Server shutting down...")
                break
            except Exception as e:
                print(f"Error handling Unity client: {e}")
                import traceback

                traceback.print_exc()
                continue

    env.close()
    print("PPO Server stopped.")


if __name__ == "__main__":
    main()
