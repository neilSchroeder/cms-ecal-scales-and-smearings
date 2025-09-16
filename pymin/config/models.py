from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from .defaults import UNSET
from .prune_config import PruneConfig


@dataclass
class InputConfig:
    """
    Configuration class for input file paths.
    This class defines the file paths for various input files used in the
    CMS ECAL scales and smearings analysis.

    Attributes
    ----------
    data_files : str, optional
        Path to the file(s) containing the data dataset.
    mc_files : str, optional
        Path to the file(s) containing the MC dataset.
    categories_file : str, optional
        Path to the file defining event categories.
    scales_file : str, optional
        Path to the file containing energy scale corrections.
    smearings_file : str, optional
        Path to the file containing energy smearing parameters.
    weights_file : str, optional
        Path to the file containing event weights.
    """

    data_files: Optional[str] = UNSET
    mc_files: Optional[str] = UNSET
    categories_file: Optional[str] = UNSET
    scales_file: Optional[str] = UNSET
    smearings_file: Optional[str] = UNSET
    weights_file: Optional[str] = UNSET


@dataclass
class WorkflowConfig:
    """
    Configuration class for workflow execution parameters.
    This class defines boolean flags that control various aspects of the workflow
    execution, including data processing steps and output generation.

    Attributes
    ----------
    prune : bool, default False
        Flag to enable the pruning operation in the workflow.
    run_divide : bool, default False
        Flag to enable the division operation in the workflow.
    time_stability : bool, default False
        Flag to enable time stability analysis.
    closure : bool, default False
        Flag to enable closure test execution.
    plot : bool, default False
        Flag to enable plot generation and visualization output.

    Examples
    --------
    >>> config = WorkflowConfig()
    >>> config.run_divide = True
    >>> config.plot = True
    """

    prune: bool = False
    run_divide: bool = False
    time_stability: bool = False
    closure: bool = False
    plot: bool = False


@dataclass
class RunDivideConfig:
    """Configuration for run division and time binning.

    Attributes
    ----------
    enabled : bool
        Whether to perform run division.
    output_file : str, optional
        Path for run division output file.
    """

    enabled: bool = False
    output_file: str | None = None


@dataclass
class TimeStabilityConfig:
    """Configuration for time stability corrections.

    Attributes
    ----------
    enabled : bool
        Whether to apply time stability corrections.
    run_bins_file : str, optional
        Path to file containing run bins for time stability.
    """

    enabled: bool = False
    run_bins_file: str | None = None


@dataclass
class MinimizationConfig:
    """
    Configuration class for minimization parameters in CMS ECAL scales and smearings analysis.
    This class defines the configuration parameters used for histogram binning, scanning ranges,
    and minimization behavior in the ECAL calibration process.

    Attributes
    ----------
    hist_min : float, default 80.0
        Minimum value for histogram range.
    hist_max : float, default 100.0
        Maximum value for histogram range.
    bin_size : float, default 0.25
        Size of histogram bins.
    auto_bin : bool, default True
        Whether to automatically determine bin size.
    start_style : str, default "scan"
        Starting style for minimization process.
    scan_min : float, default 0.98
        Minimum value for parameter scanning range.
    scan_max : float, default 1.02
        Maximum value for parameter scanning range.
    scan_step : float, default 0.001
        Step size for parameter scanning.
    fix_scales : bool, default False
        Whether to fix scale parameters during minimization.
    """

    hist_min: float = 80.0
    hist_max: float = 100.0
    bin_size: float = 0.25
    auto_bin: bool = True
    start_style: str = "scan"
    scan_min: float = 0.98
    scan_max: float = 1.02
    scan_step: float = 0.001
    fix_scales: bool = False


@dataclass
class PlottingConfig:
    """
    Configuration class for plotting settings.
    This class manages plotting-related configuration parameters including
    figure size, DPI, and style.

    Attributes
    ----------
    fig_size : tuple, default (10, 8)
        Size of the figure in inches.
    dpi : int, default 100
        Dots per inch for the figure resolution.
    style : str, default "ggplot"
        Matplotlib style to use for plots.
    """

    style: str = UNSET
    lumi_label: Optional[str] = UNSET


@dataclass
class OutputConfig:
    """
    Configuration class for output settings.
    This class manages output-related configuration parameters including
    tagging, plot directory location, and luminosity labeling.

    Attributes
    ----------
    tag : str, optional
        Optional tag string for identifying or labeling outputs.
        Default is UNSET.
    plot_dir : str
        Directory path where plots will be saved.
        Default is "./".
    lumi_label : str, optional
        Optional luminosity label for plot annotations.
        Default is UNSET.
    """

    tag: Optional[str] = UNSET
    plot_dir: str = "./"


@dataclass
class CondorConfig:
    """
    Configuration class for Condor job submission settings.
    This class manages the configuration parameters required for submitting
    jobs to HTCondor, a distributed computing system.
    Attributes
    ----------
    enabled : bool, default False
        Flag indicating whether Condor submission is enabled.
    queue : str, default "tomorrow"
        The name of the Condor queue to submit jobs to.
    Examples
    --------
    >>> config = CondorConfig()
    >>> config.enabled = True
    >>> config.queue = "short"
    """

    enabled: bool = False
    queue: str = "tomorrow"


@dataclass
class AdvancedConfig:
    """Configuration for advanced options and experimental features.

    Attributes
    ----------
    debug_mode : bool
        Enable debug output and verbose logging.
    experimental_features : bool
        Enable experimental features (use with caution).
    custom_options : dict
        Dictionary for additional custom configuration options.
    """

    debug_mode: bool = False
    experimental_features: bool = False
    custom_options: dict = field(default_factory=dict)


@dataclass
class PyMinConfig:
    """Main configuration class for PyMin calibration workflow.

    This class contains all configuration sections needed for the complete
    CMS ECAL calibration workflow, from input data handling through final
    output generation.

    Attributes
    ----------
    input : InputConfig
        Configuration for input data handling and file paths.
    workflow : WorkflowConfig
        Configuration for workflow execution and control flow.
    prune : PruneConfig
        Configuration for data pruning operations.
    run_divide : RunDivideConfig
        Configuration for run division and time binning.
    time_stability : TimeStabilityConfig
        Configuration for time stability corrections.
    minimization : MinimizationConfig
        Configuration for minimization algorithms and parameters.
    plotting : PlottingConfig
        Configuration for plotting and visualization.
    output : OutputConfig
        Configuration for output formatting and file generation.
    condor : CondorConfig
        Configuration for HTCondor job submission and management.
    advanced : AdvancedConfig
        Configuration for advanced options and experimental features.

    Examples
    --------
    >>> config = PyMinConfig()
    >>> config.input.data_path = "/path/to/data"
    >>> config.minimization.algorithm = "BFGS"
    """

    input: InputConfig = field(default_factory=InputConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    run_divide: RunDivideConfig = field(default_factory=RunDivideConfig)
    time_stability: TimeStabilityConfig = field(default_factory=TimeStabilityConfig)
    minimization: MinimizationConfig = field(default_factory=MinimizationConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    condor: CondorConfig = field(default_factory=CondorConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
