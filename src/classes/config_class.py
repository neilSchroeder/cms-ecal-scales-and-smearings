import os


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

        # set up the directories
        self.is_on_eos = os.path.exists("/eos/")
        self.set_up_directories()
        """
        sets up the following variables:
        
        self.DEFAULT_EOS_PATH
        self.DEFAULT_DATA_PATH
        self.DEFAULT_PLOT_PATH
        self.DEFAULT_WRITE_FILES_PATH
        self.DEFAULT_CONDOR_PATH
        """

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

    def set_up_directories(self, eos=True):
        """
        Set up the directories for the scales and smearings framework
        """

        # make the directory for the dat files
        self.DEFAULT_WRITE_FILES_PATH = "datFiles/"
        if not os.path.exists(self.DEFAULT_WRITE_FILES_PATH):
            os.makedirs(self.DEFAULT_WRITE_FILES_PATH, exist_ok=True)

        # make the directory for the eos files
        self.DEFAULT_EOS_PATH = self.configure_default_eos_path()
        if not os.path.exists(self.DEFAULT_EOS_PATH):
            os.makedirs(self.DEFAULT_EOS_PATH, exist_ok=True)

        # make the directory for the plots
        self.DEFAULT_PLOT_PATH = self.configure_default_plot_path()
        if not os.path.exists(self.DEFAULT_PLOT_PATH):
            os.makedirs(self.DEFAULT_PLOT_PATH, exist_ok=True)

        # make the directory for the condor files
        self.DEFAULT_CONDOR_PATH = self.configure_default_condor_path()
        if not os.path.exists(self.DEFAULT_CONDOR_PATH):
            os.makedirs(self.DEFAULT_CONDOR_PATH, exist_ok=True)

        # make the directory for the data files
        self.DEFAULT_DATA_PATH = self.configure_default_data_path()
        if not os.path.exists(self.DEFAULT_DATA_PATH):
            os.makedirs(self.DEFAULT_DATA_PATH, exist_ok=True)
