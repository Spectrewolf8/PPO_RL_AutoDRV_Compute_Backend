# connection_manager.py
import json
import zmq
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
    ZeroMQ-based connection manager using REP/REQ pattern.
    Completely independent and reusable for any ZeroMQ REP/REQ communication.
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 65432, timeout_ms: int = 1000
    ):
        """
        Initialize the ConnectionManager with ZeroMQ.

        Args:
            host: Server host address
            port: Server port number
            timeout_ms: Socket timeout in milliseconds (for non-blocking operations)
        """
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._context = None
        self._socket = None
        self._state = ConnectionState.DISCONNECTED
        self._endpoint = f"tcp://{host}:{port}"
        self._last_client_identity = None

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if a client is currently connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def client_address(self) -> Optional[str]:
        """Get the connected client's identity/address."""
        return self._last_client_identity

    def create_server(self) -> bool:
        """
        Create and bind the ZeroMQ REP socket.

        Returns:
            True if server created successfully, False otherwise
        """
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REP)

            # Set socket options
            self._socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
            self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)  # Receive timeout

            self._socket.bind(self._endpoint)
            self._state = ConnectionState.LISTENING
            return True
        except Exception as e:
            self._state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to create server: {e}")

    def accept_client(self) -> bool:
        """
        Wait for an incoming client connection (blocking with timeout).
        In ZeroMQ REP/REQ pattern, this waits for the first message.

        Returns:
            True if client message received (connection established), False on timeout
        """
        if self._state != ConnectionState.LISTENING:
            return False

        try:
            # ZeroMQ REP/REQ doesn't have explicit "accept" - we check for first message
            # This is non-blocking due to RCVTIMEO
            _ = self._socket.recv(zmq.NOBLOCK)

            # If we received data, we need to send a response (REP/REQ pattern requirement)
            # We'll store this message to be processed later
            self._pending_message = _
            self._state = ConnectionState.CONNECTED
            self._last_client_identity = (
                self._endpoint
            )  # ZeroMQ doesn't expose client address in REP/REQ
            return True
        except zmq.Again:
            # Timeout - no message received yet
            return False
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
            # Check if we have a pending message from accept_client
            if hasattr(self, "_pending_message"):
                data = self._pending_message
                delattr(self, "_pending_message")
                return data

            # Set a longer timeout for normal operation
            self._socket.setsockopt(
                zmq.RCVTIMEO, -1
            )  # No timeout during active connection
            data = self._socket.recv()

            if not data:
                # Empty message - connection closed
                self._handle_disconnect()
                return None

            return data

        except zmq.Again:
            # Timeout
            return None
        except (zmq.ZMQError, Exception):
            # Connection error
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
            self._socket.send(data)
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
        """Close the server socket and context."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        if self._context:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None

        self._state = ConnectionState.DISCONNECTED

    def _handle_disconnect(self) -> None:
        """Internal method to handle client disconnection."""
        self._last_client_identity = None

        if self._socket and self._context:
            # In ZeroMQ REP/REQ, we keep the socket open for new connections
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
