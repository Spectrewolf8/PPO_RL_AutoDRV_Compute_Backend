# connection_manager.py
import json
import socket
from typing import Optional
from enum import Enum


class ConnectionState(Enum):
    """Enum representing connection states."""

    DISCONNECTED = 0
    LISTENING = 1
    CONNECTED = 2
    ERROR = 3


class ConnectionManager:
    """
    Generic socket connection manager.
    Completely independent and reusable for any TCP socket communication.
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 65432, buffer_size: int = 4096
    ):
        """
        Initialize the ConnectionManager.

        Args:
            host: Server host address
            port: Server port number
            buffer_size: Size of receive buffer in bytes
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self._server_socket = None
        self._client_socket = None
        self._client_address = None
        self._state = ConnectionState.DISCONNECTED

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if a client is currently connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def client_address(self) -> Optional[tuple]:
        """Get the connected client's address."""
        return self._client_address

    def create_server(self) -> bool:
        """
        Create and bind the server socket.

        Returns:
            True if server created successfully, False otherwise
        """
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen()
            self._state = ConnectionState.LISTENING
            return True
        except Exception as e:
            self._state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to create server: {e}")

    def accept_client(self) -> bool:
        """
        Accept an incoming client connection (blocking).

        Returns:
            True if client accepted successfully, False otherwise
        """
        if self._state != ConnectionState.LISTENING:
            return False

        try:
            self._client_socket, self._client_address = self._server_socket.accept()
            self._state = ConnectionState.CONNECTED
            return True
        except Exception as e:
            self._state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to accept client: {e}")

    def receive_raw(self) -> Optional[bytes]:
        """
        Receive raw bytes from the client.

        Returns:
            Received bytes, or None if connection closed or error occurred
        """
        if not self.is_connected:
            return None

        try:
            data = self._client_socket.recv(self.buffer_size)

            if not data:
                # Connection closed by client
                self._handle_disconnect()
                return None

            return data

        except (ConnectionResetError, BrokenPipeError, socket.error):
            self._handle_disconnect()
            return None
        except Exception:
            self._handle_disconnect()
            return None

    def receive_json(self) -> Optional[dict]:
        """
        Receive and parse JSON data from the client.

        Returns:
            Parsed JSON as dictionary, or None if error occurred
        """
        data = self.receive_raw()

        if data is None:
            return None

        try:
            return json.loads(data.decode())
        except json.JSONDecodeError:
            return None

    def send_raw(self, data: bytes) -> bool:
        """
        Send raw bytes to the client.

        Args:
            data: Bytes to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_connected:
            return False

        try:
            self._client_socket.sendall(data)
            return True
        except Exception:
            self._handle_disconnect()
            return False

    def send_json(self, data: dict) -> bool:
        """
        Send JSON data to the client.

        Args:
            data: Dictionary to send as JSON

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            json_bytes = json.dumps(data).encode()
            return self.send_raw(json_bytes)
        except Exception:
            return False

    def send_string(self, message: str) -> bool:
        """
        Send a string message to the client.

        Args:
            message: String message to send

        Returns:
            True if sent successfully, False otherwise
        """
        return self.send_raw(message.encode())

    def disconnect_client(self) -> None:
        """Disconnect the current client."""
        self._handle_disconnect()

    def close_server(self) -> None:
        """Close the server socket and any active connections."""
        self.disconnect_client()

        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None

        self._state = ConnectionState.DISCONNECTED

    def _handle_disconnect(self) -> None:
        """Internal method to handle client disconnection."""
        if self._client_socket:
            try:
                self._client_socket.close()
            except Exception:
                pass
            self._client_socket = None

        self._client_address = None

        if self._server_socket:
            self._state = ConnectionState.LISTENING
        else:
            self._state = ConnectionState.DISCONNECTED

    def __enter__(self):
        """Context manager entry."""
        self.create_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_server()
