import os

def configure_default_eos_path():
    """ Get the default EOS path for the scales and smearings framework """
    user = os.environ['USER']
    eos_path = f'/eos/home-{user[0]}/{user}/pymin/'
    if not os.path.exists(eos_path):
        os.makedirs(eos_path, exist_ok=True)
    return eos_path


def configure_default_data_path():
    """ Get the default data path for the scales and smearings framework """
    user = os.environ['USER']
    data_path = f'/eos/home-{user[0]}/{user}/pymin/data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    return data_path


def configure_default_plot_path():
    """ Get the default plot path for the scales and smearings framework """
    user = os.environ['USER']
    plot_path = f'/eos/home-{user[0]}/{user}/pymin/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)
    return plot_path

def configure_default_condor_path():
    """ Get the default condor path for the scales and smearings framework """
    user = os.environ['USER']
    condor_path = f'/eos/home-{user[0]}/{user}/pymin/condor/'
    if not os.path.exists(condor_path):
        os.makedirs(condor_path, exist_ok=True)
    return condor_path


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
    