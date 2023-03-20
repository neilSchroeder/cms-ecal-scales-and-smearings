import python.classes.config_class as config_class

def test_ss_config():
    config = config_class.SSConfig()
    print(config.DEFAULT_DATA_PATH)
    print(config.DEFAULT_EOS_PATH)
    print(config.DEFAULT_PLOT_PATH)
    print(config.DEFAULT_WRITE_FILES_PATH)
    print(config.DEFAULT_CONDOR_PATH)

if __name__ == '__main__':
    test_ss_config()
    