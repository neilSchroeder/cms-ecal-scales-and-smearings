from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from .defaults import UNSET


@dataclass
class TreeConfig:
    """
    Configuration for ROOT tree and branch selection.

    Attributes
    ----------
    tree_name : str, default "selected"
        Name of the ROOT tree to read from.
    branches : List[str]
        List of branch names to extract from the ROOT tree.
        Default includes essential branches for ECAL calibration.
    """

    tree_name: str = "selected"
    branches: List[str] = field(
        default_factory=lambda: [
            "R9Ele",
            "energy_ECAL_ele",
            "etaEle",
            "phiEle",
            "gainSeedSC",
            "invMass_ECAL_ele",
            "runNumber",
            "eleID",
        ]
    )


@dataclass
class SelectionConfig:
    """
    Configuration for event selection cuts applied during pruning.

    Attributes
    ----------
    r9_min : float, default 0.0
        Minimum R9 value for electron selection.
    r9_max : float, default 1.0
        Maximum R9 value for electron selection.
    eta_max : float, default 2.5
        Maximum |eta| value for electron selection.
    et_min : float, default 25.0
        Minimum ET value for electron selection (GeV).
    mass_min : float, default 60.0
        Minimum invariant mass for Z peak analysis (GeV).
    mass_max : float, default 120.0
        Maximum invariant mass for Z peak analysis (GeV).
    """

    r9_min: float = 0.0
    r9_max: float = 1.0
    eta_max: float = 2.5
    et_min: float = 25.0
    mass_min: float = 60.0
    mass_max: float = 120.0


@dataclass
class ProcessingOptions:
    """
    Configuration for processing and memory management options.

    Attributes
    ----------
    chunk_size : int, default 100000
        Number of events to process in each chunk for memory management.
    compress : bool, default True
        Whether to compress output TSV files.
    verbose : bool, default True
        Whether to enable verbose logging during processing.
    """

    chunk_size: int = 100000
    compress: bool = True
    verbose: bool = True


@dataclass
class OutputFormat:
    """
    Configuration for output file formatting.

    Attributes
    ----------
    format : str, default "tsv"
        Output file format (tsv recommended for memory efficiency).
    data_prefix : str, default "data_pruned"
        Prefix for data output files.
    mc_prefix : str, default "mc_pruned"
        Prefix for MC output files.
    output_dir : str, default "pruned_files/"
        Directory for pruned output files.
    """

    format: str = "tsv"
    data_prefix: str = "data_pruned"
    mc_prefix: str = "mc_pruned"
    output_dir: str = "pruned_files/"


@dataclass
class PruneConfig:
    """
    Configuration class for pruning parameters.

    This class defines comprehensive configuration for the pruning step
    in the CMS ECAL scales and smearings analysis. Pruning converts
    ROOT files to TSV format for memory efficiency.

    Attributes
    ----------
    prune_enabled : bool, default False
        Flag to enable or disable the pruning operation.
    tree_config : TreeConfig
        Configuration for ROOT tree and branch selection.
    selection : SelectionConfig, optional
        Optional event selection cuts to apply during pruning.
        If None, no cuts are applied.
    options : ProcessingOptions
        Processing and memory management options.
    output_format : OutputFormat
        Output file formatting configuration.

    Examples
    --------
    >>> prune_config = PruneConfig()
    >>> prune_config.prune_enabled = True
    >>> prune_config.tree_config.branches.append("chargeEle")
    >>> prune_config.selection.et_min = 30.0

    Notes
    -----
    The default branch selection includes essential variables for
    CMS ECAL electron energy calibration:
    - runNumber: For time stability analysis
    - R9Ele: Shower shape variable for categorization
    - etaEle: Pseudorapidity for detector region binning
    - energy_ECAL_ele: Electron energy in ECAL (calibration target)
    - invMass_ECAL_ele: Invariant mass for Z peak fits
    - etEle: Transverse energy for kinematic cuts
    - gainEle: Gain information for systematic studies
    """

    prune_enabled: bool = False
    tree_config: TreeConfig = field(default_factory=TreeConfig)
    selection: Optional[SelectionConfig] = field(default_factory=SelectionConfig)
    options: ProcessingOptions = field(default_factory=ProcessingOptions)
    output_format: OutputFormat = field(default_factory=OutputFormat)

    # Legacy compatibility
    branches: Optional[List[str]] = None  # Deprecated, use tree_config.branches

    def __post_init__(self):
        """Post-initialization to handle legacy compatibility."""
        # Support legacy branches parameter
        if self.branches is not None:
            self.tree_config.branches = self.branches
