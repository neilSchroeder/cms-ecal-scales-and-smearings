
from python.plotters.plot_run_stability import plot_run_stability

def test_plot_run_stability():
    plot_run_stability('python/tests/data/run_stability.txt', 'test', '35.9 fb^{-1} (13 TeV) 2016', corrected=False)
    plot_run_stability('python/tests/data/run_stability.txt', 'test', '35.9 fb^{-1} (13 TeV) 2016', corrected=True)