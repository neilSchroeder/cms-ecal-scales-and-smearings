import os

def configure_default_eos_path():
    """ Get the default EOS path for the scales and smearings framework """
    user = os.environ['USER']
    return f'/eos/home-{user[0]}/{user}/pymin/'


def configure_default_data_path():
    """ Get the default data path for the scales and smearings framework """
    user = os.environ['USER']
    return f'/eos/home-{user[0]}/{user}/pymin/data/'


def configure_default_plot_path():
    """ Get the default plot path for the scales and smearings framework """
    user = os.environ['USER']
    return f'/eos/home-{user[0]}/{user}/pymin/plots/'


def configure_default_condor_path():
    """ Get the default condor path for the scales and smearings framework """
    user = os.environ['USER']
    return f'/eos/home-{user[0]}/{user}/pymin/condor/'


def set_up_directories():
    """
    Set up the directories for the scales and smearings framework
    """
    # make the directory for the dat files
    if not os.path.exists(SSConfig.DEFAULT_WRITE_FILES_PATH):
        os.makedirs(SSConfig.DEFAULT_WRITE_FILES_PATH, exist_ok=True)

    # make the directory for the plots
    if not os.path.exists(SSConfig.DEFAULT_PLOT_PATH):
        os.makedirs(SSConfig.DEFAULT_PLOT_PATH, exist_ok=True)

    # make the directory for the condor files
    if not os.path.exists(SSConfig.DEFAULT_CONDOR_PATH):
        os.makedirs(SSConfig.DEFAULT_CONDOR_PATH, exist_ok=True)

    # make the directory for the data files
    if not os.path.exists(SSConfig.DEFAULT_DATA_PATH):
        os.makedirs(SSConfig.DEFAULT_DATA_PATH, exist_ok=True)

    # make the directory for the eos files
    if not os.path.exists(SSConfig.DEFAULT_EOS_PATH):
        os.makedirs(SSConfig.DEFAULT_EOS_PATH, exist_ok=True)


class SSConfig(object):
    """ Configuration class for the scales and smearings framework """

    # this class is a singleton
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SSConfig, cls).__new__(cls)
        return cls.instance

    DEFAULT_EOS_PATH = configure_default_eos_path()
    DEFAULT_DATA_PATH = configure_default_data_path()
    DEFAULT_PLOT_PATH = configure_default_plot_path()
    DEFAULT_WRITE_FILES_PATH = "datFiles/"
    DEFAULT_CONDOR_PATH = configure_default_condor_path()

    set_up_directories()
    