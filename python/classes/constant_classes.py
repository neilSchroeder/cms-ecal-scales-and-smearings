
class PyValConstants():
    """a useful list of contants instead of defining them in every function"""

    ETA_LEAD = "eta0"
    ETA_SUB = "eta1"
    R9_LEAD = "r90"
    R9_SUB = "r91"
    ET_LEAD = "et0"
    ET_SUB = "et1"

    KEY_DAT = "DATA"
    KEY_MC = "MC"
    KEY_SC = "SCALES"
    KEY_SM = "SMEARINGS"
    KEY_WT = "WEIGHTS"
    KEY_CAT = "CATS"

    KEY_INVMASS_UP = "invmass_up"
    KEY_INVMASS_DOWN = "invmass_down"
    KEY_PTY = "pty_weight"
    
    plotting_functions = {}

    def get_plotting_function(self, style):
        if style in self.plotting_functions.keys():
            return self.plotting_functions[style]
        else:
            print(f'[ERROR] style {style} does not have a corresponding plotting function')
            print(f'[ERROR] please either define it or check your config file')
            return None
        

class DataConstants():
    """a useful list of contants instead of defining them in every function"""

    # pruning constants
    KEEP_COLS = ['R9Ele', 'energy_ECAL_ele', 'etaEle', 'phiEle', 'gainSeedSC', 'invMass_ECAL_ele', 'runNumber']
    DROP_LIST = ['R9Ele[2]', 'energy_ECAL_ele[2]', 'etaEle[2]', 'phiEle[2]', 'gainSeedSC[2]']

    # time stability constants
    TIME_STABILITY_HEADERS = ['run_min', 'run_max', 'eta_min', 'eta_max', 'median', 'mean', 'sigma', 'scale', 'median_corr', 'mean_corr', 'sigma_corr', 'events']


    #constants
    MIN_ET = 0
    MAX_ET = 14000
    MAX_EB = 1.4442
    MIN_EE = 1.566
    MAX_EE = 2.5
    TRACK_MAX = 1000
    MIN_PT_LEAD = 32
    MIN_PT_SUB = 20
    invmass_min = 60 #python/helpers/helper_main.py
    invmass_max = 120 #python/helpers/helper_main.py
    MIN_INVMASS, MAX_INVMASS = 80, 100
    MIN_ET_LEAD, MAX_ET_LEAD = 32, 14000
    MIN_ET_SUB, MAX_ET_SUB = 20, 14000
    MIN_E, MAX_E = 0, 14000

    #dataframe keys
    RUN = 'runNumber'
    ETA_LEAD = 'etaEle[0]'
    ETA_SUB = 'etaEle[1]'
    PHI_LEAD = 'phiEle[0]'
    PHI_SUB = 'phiEle[1]'
    R9_LEAD = 'R9Ele[0]'
    R9_SUB = 'R9Ele[1]'
    E_LEAD = 'energy_ECAL_ele[0]'
    E_SUB = 'energy_ECAL_ele[1]'
    GAIN_LEAD = 'gainSeedSC[0]'
    GAIN_SUB = 'gainSeedSC[1]'
    INVMASS = 'invMass_ECAL_ele'
    ET_LEAD = 'transverse_energy[0]'
    ET_SUB = 'transverse_energy[1]'

    KEEP_COLS = []

    # indices of row in scales
    i_run_min = 0
    i_run_max = 1
    i_eta_min = 2
    i_eta_max = 3
    i_r9_min = 4
    i_r9_max = 5
    i_et_min = 6
    i_et_max = 7
    i_gain = 8
    i_scale = 9
    i_err = 10

    # indices of row in plot cats
    i_plot_style = 0
    i_plot_name = 1
    i_plot_var = 2
    i_plot_bounds_eta_lead = 3
    i_plot_bounds_eta_sub = 4
    i_plot_bounds_r9_lead = 5
    i_plot_bounds_r9_sub = 6
    i_plot_bounds_et_lead = 7
    i_plot_bounds_et_sub = 8

    SEED = 3543136929

    # constants for pyt weight files
    PTY_WEIGHT_HEADERS = ['y_min', 'y_max', 'pt_min', 'pt_max', 'weight']
    YMIN = 'y_min'
    YMAX = 'y_max'
    PTMIN = 'pt_min'
    PTMAX = 'pt_max'
    WEIGHT = 'weight'
    PTZ = 'ptZ'
    RAPIDITY = 'rapidity'
    PTY_WEIGHT = 'pty_weight'
    PTZ_BINS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,39,40,45,50,55,60,80,100,14000]
    YZ_BINS = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]


class CategoryConstants():
    """ constants for the categories files """

    # categories should have the form: type etaMin etaMax r9Min r9Max gain etMin etMax
    i_type = 0
    i_eta_min = 1
    i_eta_max = 2
    i_r9_min = 3
    i_r9_max = 4
    i_gain = 5
    i_et_min = 6
    i_et_max = 7

    empty = -1
