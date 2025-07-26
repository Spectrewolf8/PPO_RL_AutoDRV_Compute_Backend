import json
import socket
import time
import random

HOST = "127.0.0.1"  # Listen on all available interfaces
PORT = 65432  # Port to listen on (matches Unity client)


def generate_ppo_action(game_state):
    """
    Generate a PPO action based on game state.
    This is a placeholder - replace with actual PPO model inference.
    Returns discrete steering: -1 (left) or 1 (right)
    """
    # Extract ray data for decision making
    ray_distances = game_state.get("rayDistances", [1.0, 1.0, 1.0, 1.0, 1.0])
    ray_hits = game_state.get("rayHits", [0, 0, 0, 0, 0])

    # Simple rule-based logic (replace with PPO model)
    steering = 1  # Default to right

    # Check forward ray (index 0)
    if ray_hits[0] == 1 and ray_distances[0] < 2.0:  # Obstacle ahead
        # Turn away from the closest side obstacle
        if ray_hits[3] == 1 and ray_distances[3] < ray_distances[4]:  # Right obstacle closer
            steering = -1  # Turn left
        elif ray_hits[4] == 1:  # Left obstacle
            steering = 1  # Turn right
        else:
            steering = random.choice([-1, 1])  # Random discrete turn

    # Fine adjustments based on side rays
    elif ray_hits[3] == 1 and ray_distances[3] < 1.5:  # Right obstacle close
        steering = -1  # Turn left
    elif ray_hits[4] == 1 and ray_distances[4] < 1.5:  # Left obstacle close
        steering = 1  # Turn right
    else:
        # Random exploration with discrete values
        steering = random.choice([-1, 1])

    return {"steering": steering}


def print_game_state_summary(game_state):
    """Print a summary of the received game state"""
    rewards = game_state.get("rewards", 0)
    respawns = game_state.get("respawns", 0)
    elapsed_time = game_state.get("elapsedTime", 0)
    car_speed = game_state.get("carSpeed", 0)

    # Ray information
    ray_distances = game_state.get("rayDistances", [])
    ray_hits = game_state.get("rayHits", [])

    print(f"Game State - Rewards: {rewards}, Respawns: {respawns}, Time: {elapsed_time:.1f}s, Speed: {car_speed:.2f}")

    if len(ray_distances) >= 5:
        ray_names = ["Forward", "Fwd-Left", "Fwd-Right", "Right", "Left"]
        for i, (name, dist, hit) in enumerate(zip(ray_names, ray_distances, ray_hits)):
            status = "HIT" if hit else "CLEAR"
            print(f"  {name}: {dist:.2f} ({status})")


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
                print(f"Unity client connected from {addr}")

                while True:
                    try:
                        data = conn.recv(4096)  # Increased buffer for game state data
                        if not data:
                            print(f"Unity client {addr} disconnected normally")
                            break

                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"

                        try:
                            # Parse the incoming message
                            message_data = json.loads(data.decode())
                            message_type = message_data.get("message", "unknown")
                            message_id = message_data.get("id", 0)
                            game_state = message_data.get("gameState", {})

                            print(f"\n--- Message {message_id} at {timestamp} ---")
                            print_game_state_summary(game_state)

                            if message_type == "game_state":
                                # Generate PPO action based on game state
                                action = generate_ppo_action(game_state)

                                # Send action response
                                response_data = {"steering": action["steering"]}

                                response = json.dumps(response_data).encode()
                                steering_direction = "LEFT" if action["steering"] == -1 else "RIGHT"
                                print(f"Sending action - Steering: {action['steering']} ({steering_direction})")
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
                        print(f"Unity client {addr} disconnected unexpectedly (connection reset)")
                        break
                    except BrokenPipeError:
                        print(f"Unity client {addr} disconnected (broken pipe)")
                        break
                    except socket.error as e:
                        if e.errno == 104:  # Connection reset by peer
                            print(f"Unity client {addr} reset the connection")
                        else:
                            print(f"Socket error with Unity client {addr}: {e}")
                        break

        except KeyboardInterrupt:
            print("\nPPO Server shutting down...")
            break
        except Exception as e:
            print(f"Error handling Unity client: {e}")
            continue

print("PPO Server stopped.")
