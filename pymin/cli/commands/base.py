"""
Base class for CLI commands
"""

from abc import ABC, abstractmethod
import logging
from typing import List

from pymin.config.models import PyMinConfig
from pymin.config.defaults import UNSET


class BaseCommand(ABC):
    """
    Base class for CLI commands

    Attributes:
        config (PyMinConfig): Configuration object
        logger (logging.Logger): Logger instance

    Methods:
        execute() -> bool: Execute the command and return success status
        validate() -> List[str]: Validate inputs and return list of errors
    """

    def __init__(self, config: PyMinConfig, logger: logging.Logger = UNSET):
        self.config = config
        if logger is not UNSET:
            self.logger = logger
        else:

            self.logger = logging.getLogger(__name__)

    @abstractmethod
    def execute(self) -> bool:
        """Execute the command and return success status"""

    @abstractmethod
    def validate(self) -> List[str]:
        """Validate inputs and return list of errors"""
