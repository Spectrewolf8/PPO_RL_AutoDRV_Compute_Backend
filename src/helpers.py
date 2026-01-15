import logging
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DebugHelper:
    """Helper class for logging and debugging."""

    _logger = logging.getLogger(__name__)

    @staticmethod
    def log(message: str) -> None:
        """Log an info message."""
        DebugHelper._logger.info(message)

    @staticmethod
    def log_error(error_message: str) -> None:
        """Log an error message."""
        DebugHelper._logger.error(error_message)

    @staticmethod
    def warn(warning_message: str) -> None:
        """Log a warning message."""
        DebugHelper._logger.warning(warning_message)

    @staticmethod
    def debug(debug_message: str) -> None:
        """Log a debug message."""
        DebugHelper._logger.debug(debug_message)

    @staticmethod
    def print_game_state_summary(game_state: Dict[str, Any]) -> None:
        """Print a summary of the received game state."""
        rewards = game_state.get("rewards", 0)
        respawns = game_state.get("respawns", 0)
        elapsed_time = game_state.get("elapsedTime", 0)
        car_speed = game_state.get("carSpeed", 0)

        # Ray information
        ray_distances = game_state.get("rayDistances", [])
        ray_hits = game_state.get("rayHits", [])

        summary = (
            f"Game State - Rewards: {rewards}, Respawns: {respawns}, "
            f"Time: {elapsed_time:.1f}s, Speed: {car_speed:.2f}"
        )
        DebugHelper._logger.info(summary)

        if len(ray_distances) >= 5:
            ray_names = ["Forward", "Fwd-Left", "Fwd-Right", "Right", "Left"]
            for i, (name, dist, hit) in enumerate(
                zip(ray_names, ray_distances, ray_hits)
            ):
                status = "HIT" if hit else "CLEAR"
                DebugHelper._logger.info(f"  {name}: {dist:.2f} ({status})")

    @staticmethod
    def set_level(level: int) -> None:
        """
        Set the logging level.

        Args:
            level: Logging level (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
        """
        DebugHelper._logger.setLevel(level)

    # @staticmethod
    # def add_file_handler(filename: str, level: int = logging.INFO) -> None:
    #     """
    #     Add a file handler to save logs to a file.

    #     Args:
    #         filename: Path to the log file
    #         level: Logging level for file handler
    #     """
    #     file_handler = logging.FileHandler(filename)
    #     file_handler.setLevel(level)
    #     formatter = logging.Formatter(
    #         "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    #     )
    #     file_handler.setFormatter(formatter)
    #     DebugHelper._logger.addHandler(file_handler)
