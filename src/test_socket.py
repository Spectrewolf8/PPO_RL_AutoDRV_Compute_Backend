import socket

# 100.90.237.12:65432
HOST = "100.90.237.12"
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    while True:
        try:
            s.sendall(b"Hello World")
        except KeyboardInterrupt:
            s.close()
            print("\nClient shutting down...")
            break
    data = s.recv(1024)
    print(data.decode())
