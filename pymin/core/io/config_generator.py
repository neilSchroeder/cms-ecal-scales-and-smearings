"""Configuration generator for pipeline integration."""

import logging
from pathlib import Path
import yaml

from pymin.config.defaults import UNSET


class ConfigGenerator:
    """Service for generating configuration files for pipeline steps."""

    def __init__(self, logger: logging.Logger = UNSET):
        self.logger = logger or logging.getLogger(__name__)

    def create_next_step_config(
        self, tag: str, pruned_dir: Path, data_prefix: str, mc_prefix: str
    ) -> Path:
        """Create configuration for next pipeline step.

        Parameters
        ----------
        tag : str
            Output tag for naming
        pruned_dir : Path
            Directory containing pruned files
        data_prefix : str
            Prefix for data files
        mc_prefix : str
            Prefix for MC files

        Returns
        -------
        Path
            Path to generated configuration file
        """
        # Create next step configuration
        next_config = {
            "input": {
                "data_files": [str(pruned_dir / f"{data_prefix}_data.tsv.gz")],
                "mc_files": [str(pruned_dir / f"{mc_prefix}_mc.tsv.gz")],
            },
            "output": {"tag": f"{tag}_next"},
            "workflow": {"run_divide": True},  # Next typical step
        }

        # Write configuration file
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / f"{tag}_next_step.yaml"

        with open(config_file, "w") as f:
            yaml.dump(next_config, f, default_flow_style=False, indent=2)

        self.logger.info(f"Generated next step config: {config_file}")
        return config_file
