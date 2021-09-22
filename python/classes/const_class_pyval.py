
import python.plotters.plots as plots

class const:
    """a useful list of contants instead of defining them in every function"""

    def __init__(self):

        #constants
        self.MIN_ET = 0
        self.MAX_ET = 14000
        self.MAX_EB = 1.4442
        self.MIN_EE = 1.566
        self.MAX_EE = 2.5
        self.TRACK_MAX = 1000
        self.MIN_PT_LEAD = 32
        self.MIN_PT_SUB = 20
        self.invmass_min = 60 #python/helpers/helper_main.py
        self.invmass_max = 120 #python/helpers/helper_main.py
        self.MIN_INVMASS = 80
        self.MAX_INVMASS = 100

        #dataframe keys
        self.RUN = 'runNumber'
        self.ETA_LEAD = 'etaEle[0]'
        self.ETA_SUB = 'etaEle[1]'
        self.PHI_LEAD = 'phiEle[0]'
        self.PHI_SUB = 'phiEle[1]'
        self.R9_LEAD = 'R9Ele[0]'
        self.R9_SUB = 'R9Ele[1]'
        self.E_LEAD = 'energy_ECAL_ele[0]'
        self.E_SUB = 'energy_ECAL_ele[1]'
        self.GAIN_LEAD = 'gainSeedSC[0]'
        self.GAIN_SUB = 'gainSeedSC[1]'
        self.INVMASS = 'invMass_ECAL_ele'

        #indices of row in scales
        self.i_run_min = 0
        self.i_run_max = 1
        self.i_eta_min = 2
        self.i_eta_max = 3
        self.i_r9_min = 4
        self.i_r9_max = 5
        self.i_et_min = 6
        self.i_et_max = 7
        self.i_gain = 8
        self.i_scale = 9
        self.i_err = 10

        #indices of row in plot cats
        self.i_plot_style = 0
        self.i_plot_name = 1
        self.i_plot_var = 2
        self.i_plot_bounds_eta_lead = 3
        self.i_plot_bounds_eta_sub = 4
        self.i_plot_bounds_r9_lead = 5
        self.i_plot_bounds_r9_sub = 6
        self.i_plot_bounds_et_lead = 7
        self.i_plot_bounds_et_sub = 8

        self.SEED = 3543136929

        #plot types
        self.plotting_functions = {
                'paper': plots.plot_style_paper,
                'crossCheckMC': plots.plot_style_validation_mc
                }

    def get_plotting_function(self, style):
        if style in self.plotting_functions.keys():
            return self.plotting_functions[style]
        else:
            print(f'[ERROR] style {style} does not have a corresponding plotting function')
            print(f'[ERROR] please either define it or check your config file')
            return None

