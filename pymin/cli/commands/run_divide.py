"""Modern run divide command implementation for CMS ECAL calibration workflow."""

from pathlib import Path
from typing import List, Tuple

from pymin.cli.commands.base import BaseCommand
from pymin.config.models import PyMinConfig
from pymin.core.data.run_divider import RunDivider
from pymin.core.io.polars_reader import PolarsReader
from pymin.core.io.run_bin_writer import RunBinWriter
from pymin.core.io.config_generator import ConfigGenerator


class RunDivideCommand(BaseCommand):
    """Modern command for dividing CMS runs into bins with sufficient statistics.

    This command implements the second step of the CMS ECAL calibration workflow,
    creating run bins for time stability corrections using efficient Polars-based
    data processing and modern service architecture patterns.
    """

    def get_name(self) -> str:
        """Get the name of the command."""
        return "run_divide"

    def get_description(self) -> str:
        """Get a brief description of the command."""
        return (
            "Divide runs into bins with minimum event requirements for time stability"
        )

    def validate(self) -> List[str]:
        """Validate configuration for the run divide command.

        Returns
        -------
        List[str]
            List of validation errors, empty if valid
        """
        errors = []

        # Check run divide enabled
        if not self.config.run_divide.enabled:
            errors.append("Run divide must be enabled in configuration")
            return errors

        # Validate input data files exist (need data, not MC for run binning)
        if not self.config.input.data_files:
            errors.append("Data files required for run division")
            return errors

        # Validate data files exist
        for file_path in self.config.input.data_files:
            if not Path(file_path).exists():
                errors.append(f"Data file not found: {file_path}")

        # Validate output tag for file naming
        if not self.config.output.tag:
            errors.append("Output tag required for run bin file naming")

        # Validate minimum events is reasonable
        min_events = getattr(self.config.run_divide, "min_events", 10000)
        if min_events < 1000:
            errors.append(f"Minimum events ({min_events}) too low - recommend >= 1000")

        # Validate output directory accessibility
        output_dir = Path("datFiles")  # Legacy output location
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            errors.append(f"Cannot access output directory {output_dir}: {e}")

        return errors

    def execute(self) -> int:
        """Execute the modern run divide command.

        Returns
        -------
        int
            Exit code (0 for success, 1 for failure)
        """
        self.logger.info("Starting modern run division operation")

        try:
            # Validate configuration
            validation_errors = self.validate()
            if validation_errors:
                for error in validation_errors:
                    self.logger.error("Validation: %s", error)
                return 1

            # Get configuration parameters
            min_events = getattr(self.config.run_divide, "min_events", 10000)

            self.logger.info(
                "Dividing runs with minimum event requirement: %d", min_events
            )

            # Initialize services with dependency injection
            polars_reader = PolarsReader(
                file_format="tsv", logger=self.logger  # Assuming pruned TSV input
            )

            run_divider = RunDivider(min_events=min_events, logger=self.logger)

            run_bin_writer = RunBinWriter(
                output_dir=Path("datFiles"), logger=self.logger
            )

            # Load data using efficient Polars streaming
            self.logger.info("Loading data for run analysis")
            data = polars_reader.read_files(self.config.input.data_files)

            # Perform run division with modern algorithm
            run_bins = run_divider.divide_runs(data)

            # Write run bins to legacy format for pipeline compatibility
            output_file = run_bin_writer.write_run_bins(
                run_bins=run_bins, output_tag=self.config.output.tag
            )

            # Generate configuration for next pipeline step
            if self.config.output.tag:
                self._generate_pipeline_config()

            self.logger.info("Run division completed successfully")
            self.logger.info("Created %d run bins in: %s", len(run_bins), output_file)
            return 0

        except Exception as e:
            self.logger.error("Run divide command failed: %s", e)
            if self.logger.level <= 10:  # DEBUG
                import traceback

                traceback.print_exc()
            return 1

    def _generate_pipeline_config(self) -> None:
        """Generate configuration for next pipeline step (time stability)."""
        generator = ConfigGenerator(logger=self.logger)

        # Next step uses run bins as categories for time stability
        run_bin_file = Path("datFiles") / f"run_divide_{self.config.output.tag}.dat"

        pipeline_config = generator.create_time_stability_config(
            tag=self.config.output.tag,
            input_files=self.config.input.data_files,
            run_bin_file=run_bin_file,
        )

        self.logger.info("Generated time stability config: %s", pipeline_config)
