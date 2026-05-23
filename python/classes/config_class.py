import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


class SSConfig(object):
    """Configuration class for the scales and smearings framework"""

    # this class is a singleton
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(SSConfig, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        """
        Initialize the configuration class for the scales and smearings framework
        ----------
        Params:
            DEFAULT_EOS_PATH: str
                The default EOS path for the scales and smearings framework
            DEFAULT_DATA_PATH: str
                The default data path for the scales and smearings framework
            DEFAULT_PLOT_PATH: str
                The default plot path for the scales and smearings framework
            DEFAULT_WRITE_FILES_PATH: str
                The default path for writing files for the scales and smearings framework
            DEFAULT_CONDOR_PATH: str
                The default condor path for the scales and smearings framework
        ----------
        Returns:
            None
        """
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        # configure paths without creating directories
        self.is_on_eos = os.path.exists("/eos/")
        self.DEFAULT_WRITE_FILES_PATH = "datFiles/"
        self.DEFAULT_EOS_PATH = self.configure_default_eos_path()
        self.DEFAULT_DATA_PATH = self.configure_default_data_path()
        self.DEFAULT_PLOT_PATH = self.configure_default_plot_path()
        self.DEFAULT_CONDOR_PATH = self.configure_default_condor_path()

    def configure_default_eos_path(self):
        """Get the default EOS path for the scales and smearings framework"""
        if not self.is_on_eos:
            return "workspace/pymin/"
        user = os.environ["USER"]
        return f"/eos/home-{user[0]}/{user}/pymin/"

    def configure_default_data_path(self):
        """Get the default data path for the scales and smearings framework"""
        if not self.is_on_eos:
            return "workspace/pymin/data/"
        user = os.environ["USER"]
        return f"/eos/home-{user[0]}/{user}/pymin/data/"

    def configure_default_condor_path(self):
        """Get the default condor path for the scales and smearings framework"""
        if not self.is_on_eos:
            return "workspace/pymin/condor/"
        user = os.environ["USER"]
        return f"/eos/home-{user[0]}/{user}/pymin/condor/"

    def configure_default_plot_path(self):
        """Get the default plot path for the scales and smearings framework"""
        if not self.is_on_eos:
            return "workspace/pymin/plots/"
        user = os.environ["USER"]
        return f"/eos/home-{user[0]}/{user}/pymin/plots/"

    def ensure_directories(self):
        """
        Create the directories for the scales and smearings framework.
        Call this once from entry points (pymin.py, pyval.py), not at import time.
        """

        for path in [
            self.DEFAULT_WRITE_FILES_PATH,
            self.DEFAULT_EOS_PATH,
            self.DEFAULT_PLOT_PATH,
            self.DEFAULT_CONDOR_PATH,
            self.DEFAULT_DATA_PATH,
        ]:
            os.makedirs(path, exist_ok=True)

    # backwards compatibility alias
    set_up_directories = ensure_directories


@dataclass
class MinimizationConfig:
    """Typed configuration for the minimization pipeline.

    Constructed from argparse args by helper_pymin.get_options() and threaded
    through minimizer → helper_minimizer → data_loader → zcat.
    """

    # histogram settings
    hist_min: float = 80.0
    hist_max: float = 100.0
    bin_size: float = 0.25
    auto_bin: bool = True

    # minimizer settings
    start_style: str = "scan"
    scan_min: float = 0.98
    scan_max: float = 1.02
    scan_step: float = 0.001
    min_step_size: Optional[str] = None
    # minimizer bounds (defaults reproduce historical hardcoded values)
    scale_bounds: Tuple[float, float] = (0.96, 1.04)
    smear_bounds: Tuple[float, float] = (0.0, 0.05)
    closure_scale_bounds: Tuple[float, float] = (0.99, 1.01)

    # workflow flags
    _kClosure: bool = False
    _kFixScales: bool = False
    _kPlot: bool = False
    _kTestMethodAccuracy: bool = False
    _kScanNLL: bool = False
    _kDebug: bool = False

    # file paths
    scales: Optional[str] = None
    ignore: Optional[str] = None
    plot_dir: str = "./"

    # derived counts (set during minimize setup)
    num_scales: int = 0
    num_smears: int = 0

    def to_dict(self) -> dict:
        """Convert to dict for backwards-compatible **kwargs passing."""
        return asdict(self)

    @classmethod
    def from_args(cls, args) -> "MinimizationConfig":
        """Build from an argparse Namespace."""
        return cls(
            hist_min=float(args.hist_min),
            hist_max=float(args.hist_max),
            bin_size=float(args.bin_size),
            auto_bin=not getattr(args, "_kNoAutoBin", False),
            start_style=args.start_style,
            scan_min=float(args.scan_min),
            scan_max=float(args.scan_max),
            scan_step=float(args.scan_step),
            min_step_size=args.min_step_size,
            _kClosure=args._kClosure,
            _kFixScales=args._kFixScales,
            _kPlot=args._kPlot,
            _kTestMethodAccuracy=args._kTestMethodAccuracy,
            _kScanNLL=args._kScanNLL,
            _kDebug=getattr(args, "_kDebug", False),
            scales=args.scales,
            ignore=args.ignore,
            plot_dir=args.plot_dir,
            scale_bounds=tuple(args.scale_bounds),
            smear_bounds=tuple(args.smear_bounds),
            closure_scale_bounds=tuple(args.closure_scale_bounds),
        )
