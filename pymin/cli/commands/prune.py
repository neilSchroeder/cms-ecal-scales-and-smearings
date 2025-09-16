"""Modern prune command implementation for CMS ECAL calibration workflow."""

from pathlib import Path
from typing import List

from pymin.cli.commands.base import BaseCommand
from pymin.config.models import PyMinConfig
from pymin.core.data.data_processor import DataProcessor
from pymin.core.io.root_reader import RootFileReader
from pymin.core.io.tsv_writer import TsvWriter
from pymin.core.io.config_generator import ConfigGenerator


class PruneCommand(BaseCommand):
    """Modern command for pruning ROOT files to TSV format.

    This command implements the first step of the CMS ECAL calibration workflow,
    converting ROOT files to memory-efficient TSV format using a clean service
    architecture with dependency injection and modern Python patterns.
    """

    def get_name(self) -> str:
        """Get the name of the command."""
        return "prune"

    def get_description(self) -> str:
        """Get a brief description of the command."""
        return "Convert ROOT files to TSV format with optimized memory usage"

    def validate(self) -> List[str]:
        """Validate configuration for the prune command.

        Parameters
        ----------
        config : PyMinConfig
            Complete configuration object

        Returns
        -------
        List[str]
            List of validation errors, empty if valid
        """
        errors = []

        # Check pruning enabled
        if not self.config.prune.prune_enabled:
            errors.append("Pruning must be enabled in configuration")
            return errors

        # Validate input files exist
        input_files = []
        if self.config.input.data_files:
            input_files.extend(self.config.input.data_files)
        if self.config.input.mc_files:
            input_files.extend(self.config.input.mc_files)

        if not input_files:
            errors.append("No input files specified")
            return errors

        for file_path in input_files:
            if not Path(file_path).exists():
                errors.append(f"Input file not found: {file_path}")

        # Validate required branches
        if not self.config.prune.tree_config.branches:
            errors.append("No branches specified for extraction")

        # Validate output directory accessibility
        output_dir = Path(self.config.prune.output_format.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            errors.append(f"Cannot access output directory {output_dir}: {e}")

        return errors

    def execute(self) -> int:
        """Execute the modern prune command.

        Parameters
        ----------
        config : PyMinConfig
            Complete configuration object

        Returns
        -------
        int
            Exit code (0 for success, 1 for failure)
        """
        self.logger.info("Starting modern ROOT file pruning operation")

        try:
            # Validate configuration
            validation_errors = self.validate()
            if validation_errors:
                for error in validation_errors:
                    self.logger.error("Validation: %s", error)
                return 1

            # Initialize services with dependency injection
            root_reader = RootFileReader(
                tree_name=self.config.prune.tree_config.tree_name,
                branches=self.config.prune.tree_config.branches,
                chunk_size=self.config.prune.options.chunk_size,
                logger=self.logger,
            )

            data_processor = DataProcessor(
                selection_config=self.config.prune.selection, logger=self.logger
            )

            tsv_writer = TsvWriter(
                output_dir=Path(self.config.prune.output_format.output_dir),
                compression=self.config.prune.options.compress,
                logger=self.logger,
            )

            # Process data files
            if self.config.input.data_files:
                self._process_file_group(
                    file_paths=self.config.input.data_files,
                    output_prefix=self.config.prune.output_format.data_prefix,
                    file_type="data",
                    root_reader=root_reader,
                    data_processor=data_processor,
                    tsv_writer=tsv_writer,
                )

            # Process MC files
            if self.config.input.mc_files:
                self._process_file_group(
                    file_paths=self.config.input.mc_files,
                    output_prefix=self.config.prune.output_format.mc_prefix,
                    file_type="mc",
                    root_reader=root_reader,
                    data_processor=data_processor,
                    tsv_writer=tsv_writer,
                )

            # Generate configuration for next pipeline step
            if self.config.output.tag:
                self._generate_pipeline_config(self.config)

            self.logger.info("Pruning operation completed successfully")
            return 0

        except Exception as e:
            self.logger.error("Prune command failed: %s", e)
            if self.logger.level <= 10:  # DEBUG
                import traceback

                traceback.print_exc()
            return 1

    def _process_file_group(
        self,
        file_paths: str | List[str],
        output_prefix: str,
        file_type: str,
        root_reader: "RootFileReader",
        data_processor: "DataProcessor",
        tsv_writer: "TsvWriter",
    ) -> None:
        """Process a group of ROOT files using modern service architecture.

        Parameters
        ----------
        file_paths : List[str]
            Paths to input ROOT files
        output_prefix : str
            Prefix for output file naming
        file_type : str
            Type identifier ("data" or "mc")
        root_reader : RootFileReader
            Service for reading ROOT files
        data_processor : DataProcessor
            Service for applying cuts and transformations
        tsv_writer : TsvWriter
            Service for writing TSV output
        """
        self.logger.info("Processing %d %s files", len(file_paths), file_type)

        # Use generator for memory efficiency
        data_stream = root_reader.read_files_chunked([Path(fp) for fp in file_paths])

        # Process data through pipeline
        processed_stream = data_processor.process_stream(data_stream)

        # Write output with automatic merging
        output_file = tsv_writer.write_merged_output(
            data_stream=processed_stream,
            output_prefix=output_prefix,
            file_type=file_type,
        )

        self.logger.info("Created %s output: %s", file_type, output_file)

    def _generate_pipeline_config(self, config: PyMinConfig) -> None:
        """Generate configuration for next pipeline step.

        Parameters
        ----------
        config : PyMinConfig
            Current configuration
        """

        generator = ConfigGenerator(logger=self.logger)
        pipeline_config = generator.create_next_step_config(
            tag=config.output.tag,
            pruned_dir=Path(config.prune.output_format.output_dir),
            data_prefix=config.prune.output_format.data_prefix,
            mc_prefix=config.prune.output_format.mc_prefix,
        )

        self.logger.info("Generated pipeline config: %s", pipeline_config)
