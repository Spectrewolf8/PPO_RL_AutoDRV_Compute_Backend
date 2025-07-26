import json
import socket
import time

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server listening on {HOST}:{PORT}")

    while True:  # Keep listening forever
        try:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    try:
                        data = conn.recv(1024)
                        if not data:
                            print(f"Client {addr} disconnected normally")
                            break
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
                        json_data = json.loads(data.decode())
                        print(f"Received from {addr}: {json.dumps(json_data, indent=4)} at {timestamp}")

                        # Fix: Create response as string first, then encode
                        response_str = {"message": "Acknowledged", "move": "Left", "id": json_data.get("id"), "timestamp": timestamp}
                        response = json.dumps(response_str).encode()
                        print(f"Sending to {addr}: {json.dumps(response_str, indent=4)}")
                        conn.sendall(response)
                    except ConnectionResetError:
                        print(f"Client {addr} disconnected unexpectedly (connection reset)")
                        break
                    except BrokenPipeError:
                        print(f"Client {addr} disconnected (broken pipe)")
                        break
                    except socket.error as e:
                        if e.errno == 104:  # Connection reset by peer
                            print(f"Client {addr} reset the connection")
                        else:
                            print(f"Socket error with client {addr}: {e}")
                        break
        except KeyboardInterrupt:
            print("\nServer shutting down...")
            break
        except Exception as e:
            print(f"Error handling client: {e}")
            continue
