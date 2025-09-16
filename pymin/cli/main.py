"""
Main CLI entry point for PyMin - Scales and Smearings Analysis Package.
"""

import sys
import argparse
from pathlib import Path
import logging

from pymin.config.manager import ConfigManager
from pymin.cli.commands import WORKFLOW_COMMANDS
from pymin.utils.logging import setup_logging
from pymin.constants.defaults import WORKFLOW_COMMAND_ORDER


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="pymin",
        description="PyMin: Scales and Smearings Analysis for CMS ECAL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Main operation flags
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Convert ROOT files to gzipped TSV format for memory efficiency",
    )

    parser.add_argument(
        "--run-divide",
        action="store_true",
        help="Divide data into run ranges for time stability analysis",
    )

    parser.add_argument(
        "--time-stability",
        action="store_true",
        help="Perform time stability analysis",
    )

    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Run minimization to determine scale and smearing values",
    )

    # Config file (positional-like but with validation)
    parser.add_argument(
        "config_file", nargs="?", help="Configuration file path (YAML format)"
    )

    # Common options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Increase verbosity"
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)

    # Validate arguments
    if not args.config_file:
        logger.error("Configuration file is required")
        return 1

    config_path = Path(args.config_file)
    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        return 1

    try:
        # Load configuration
        config_manager = ConfigManager(logger=logger)
        config = config_manager.load_config(config_file=str(config_path))

        # loop through args, check for values in workflow config, set if found
        for arg in args.__dict__:
            if getattr(args, arg) is True and hasattr(config.workflow, arg):
                setattr(config.workflow, arg, True)

        # build workflow based on args
        for command_name, command_class in WORKFLOW_COMMANDS.items():
            if getattr(config.workflow, command_name, False):
                command_instance = command_class(config, logger=logger)
                command_name = command_instance.get_name()
                logger.info("Executing command: %s", command_name)
                errors = command_instance.validate()
                if errors:
                    for error in errors:
                        logger.error("Validation error: %s", error)
                    return 1
                success = command_instance.execute()
                if not success:
                    logger.error("Command %s failed", command_name)
                    return 1

    except Exception as e:
        logger.error("Command failed: %s", e)
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
