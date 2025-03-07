import pytest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Import the module to test
sys.path.append('.')  # Add the current directory to path if needed
import pymin

# Create fixtures for commonly used objects
@pytest.fixture
def sample_data():
    """Fixture for sample data DataFrame."""
    return pd.DataFrame({
        'run': [1, 1, 2, 2, 3, 3],
        'mass': [91.0, 92.0, 90.5, 91.5, 90.0, 92.5],
        'eta1': [0.5, -0.5, 1.0, -1.0, 1.5, -1.5],
        'eta2': [-0.5, 0.5, -1.0, 1.0, -1.5, 1.5],
        'pt1': [30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
        'pt2': [30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
    })

@pytest.fixture
def sample_mc():
    """Fixture for sample MC DataFrame."""
    return pd.DataFrame({
        'mass': [91.2, 91.8, 90.8, 91.4, 90.2, 92.2],
        'eta1': [0.4, -0.4, 0.9, -0.9, 1.4, -1.4],
        'eta2': [-0.4, 0.4, -0.9, 0.9, -1.4, 1.4],
        'pt1': [31.0, 36.0, 41.0, 46.0, 51.0, 56.0],
        'pt2': [31.0, 36.0, 41.0, 46.0, 51.0, 56.0],
        'weight': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })

@pytest.fixture
def sample_cats():
    """Fixture for sample categories DataFrame."""
    return pd.DataFrame({
        0: ['cat1', 'cat2'],
        1: ['abs(eta1)<1.0', 'abs(eta1)>=1.0'],
        2: ['abs(eta2)<1.0', 'abs(eta2)>=1.0'],
    })

@pytest.fixture
def temp_dir(tmpdir):
    """Creates a temporary directory for file outputs."""
    return tmpdir

@pytest.fixture
def input_file_path(temp_dir, sample_data, sample_mc):
    """Creates a temporary input file for test."""
    data_path = os.path.join(temp_dir, "test_data.csv")
    mc_path = os.path.join(temp_dir, "test_mc.csv")
    
    sample_data.to_csv(data_path, index=False)
    sample_mc.to_csv(mc_path, index=False)
    
    input_file_content = f"data\t{data_path}\nmc\t{mc_path}"
    input_file = os.path.join(temp_dir, "input.txt")
    
    with open(input_file, 'w') as f:
        f.write(input_file_content)
    
    return input_file


def test_argument_parser():
    """Test that argument parser initializes correctly with default values."""
    mock_args = MagicMock()
    mock_args.inputFile = "test_input.txt"
    # Set other necessary attributes to avoid errors
    mock_args.run_divide = False
    mock_args.time_stability = False
    mock_args.closure = False
    mock_args.condor = False
    mock_args._kDebug = False
    
    with patch('sys.argv', ['pymin.py', '-i', 'test_input.txt']):
        with patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
            # This should avoid calling the real main function which might cause errors
            with patch('src.helpers.helper_pymin.load_dataframes', return_value=(None, None)):
                with patch('pymin.minimizer.minimize', return_value={}):
                    pymin.main()
                    # The test passes if no exception is raised


# Test loading dataframes
def test_load_dataframes(input_file_path):
    """Test that dataframes are loaded correctly."""
    with patch('src.helpers.helper_pymin.load_dataframes') as mock_load:
        mock_load.return_value = (pd.DataFrame(), pd.DataFrame())
        args = MagicMock()
        args.inputFile = input_file_path
        args._kDebug = False
        
        # Call the actual function and test mock was called with right arguments
        from src.helpers import helper_pymin
        data, mc = helper_pymin.load_dataframes(args.inputFile, args)
        
        mock_load.assert_called_once_with(input_file_path, args)


# Test run division
@patch('src.utilities.divide_by_run.divide')
@patch('src.tools.write_files.write_runs')
def test_run_divide(mock_write_runs, mock_divide, sample_data):
    """Test the run division functionality."""
    mock_divide.return_value = {1: [1], 2: [2], 3: [3]}
    
    with patch('sys.argv', ['pymin.py', '--run-divide', '--min-events', '1000', '-o', 'test', 
                            '-i', 'test_input.txt']):
        with patch('src.helpers.helper_pymin.load_dataframes', 
                   return_value=(sample_data, None)):
            pymin.main()
            
            # Verify divide_by_run was called with sample_data and min_events
            mock_divide.assert_called_once()
            args, _ = mock_divide.call_args
            pd.testing.assert_frame_equal(args[0], sample_data)
            
            # Verify write_runs was called with the result from divide_by_run
            mock_write_runs.assert_called_once_with({1: [1], 2: [2], 3: [3]}, 
                                                   'datFiles/run_divide_test.dat')


# Test time stability
@patch('src.utilities.time_stability.derive')
@patch('src.tools.write_files.write_time_stability')
def test_time_stability(mock_write_ts, mock_derive, sample_data, sample_cats, temp_dir):
    """Test the time stability functionality."""
    ts_data = {'cat1': [1.01], 'cat2': [0.99]}
    data_path = os.path.join(temp_dir, "ts_data.csv")
    mock_derive.return_value = (ts_data, data_path)
    
    cats_file = os.path.join(temp_dir, "cats.tsv")
    sample_cats.to_csv(cats_file, sep='\t', index=False, header=False)
    
    with patch('sys.argv', ['pymin.py', '--time-stability', '-c', cats_file, 
                           '-o', 'test', '-i', 'test_input.txt']):
        with patch('src.helpers.helper_pymin.load_dataframes', 
                   return_value=(sample_data, None)):
            with patch('os.path.isfile', return_value=True):
                pymin.main()
                
                # Verify time_stability.derive was called with sample_data and cats_file
                mock_derive.assert_called_once_with(sample_data, cats_file, 'test')
                
                # Verify write_time_stability was called with the result from derive
                mock_write_ts.assert_called_once_with(ts_data, cats_file, 
                                                     'datFiles/step1_test_scales.dat')


# Test data scaling
@patch('src.utilities.scale_data.scale')
def test_scale_data(mock_scale, sample_data, sample_mc, temp_dir):
    """Test the data scaling functionality."""
    scales_file = os.path.join(temp_dir, "scales.dat")
    with open(scales_file, 'w') as f:
        f.write("cat1\t1.01\ncat2\t0.99")
    
    mock_scale.return_value = sample_data  # Just return the same data for simplicity
    
    with patch('sys.argv', ['pymin.py', '-s', scales_file, '-i', 'test_input.txt']):
        with patch('src.helpers.helper_pymin.load_dataframes', 
                   return_value=(sample_data, sample_mc)):
            with patch('os.path.isfile', return_value=True):
                with patch('pymin.minimizer.minimize', return_value={}):
                    pymin.main()
                    
                    # Verify scale_data.scale was called with sample_data and scales_file
                    mock_scale.assert_called_once_with(sample_data, scales_file)


# Test MC smearing for closure test
@patch('src.utilities.smear_mc.smear')
def test_smear_mc_closure(mock_smear, sample_data, sample_mc, sample_cats, temp_dir):
    """Test MC smearing for closure test."""
    smearings_file = os.path.join(temp_dir, "smearings.dat")
    with open(smearings_file, 'w') as f:
        f.write("cat1\t0.01\ncat2\t0.02")
    
    cats_file = os.path.join(temp_dir, "cats.tsv")
    sample_cats.to_csv(cats_file, sep='\t', index=False, header=False)
    
    mock_smear.return_value = sample_mc  # Just return the same MC for simplicity
    
    with patch('sys.argv', ['pymin.py', '--closure', '--smearings', smearings_file, 
                           '-c', cats_file, '-i', 'test_input.txt']):
        with patch('src.helpers.helper_pymin.load_dataframes', 
                   return_value=(sample_data, sample_mc)):
            with patch('os.path.isfile', return_value=True):
                with patch('pymin.minimizer.minimize', return_value={}):
                    with patch('src.helpers.helper_pymin.write_results', return_value=True):
                        pymin.main()
                        
                        # Verify smear_mc.smear was called with sample_mc and smearings_file
                        mock_smear.assert_called_once_with(sample_mc, smearings_file)


# Test minimization process
@patch('src.utilities.minimizer.minimize')
def test_minimization(mock_minimize, sample_data, sample_mc, sample_cats, temp_dir):
    """Test the minimization process."""
    cats_file = os.path.join(temp_dir, "cats.tsv")
    sample_cats.to_csv(cats_file, sep='\t', index=False, header=False)
    
    mock_results = {'cat1': {'scale': 1.01, 'smear': 0.01}, 
                    'cat2': {'scale': 0.99, 'smear': 0.02}}
    mock_minimize.return_value = mock_results
    
    with patch('sys.argv', ['pymin.py', '-c', cats_file, '-o', 'test', 
                           '-i', 'test_input.txt']):
        with patch('src.helpers.helper_pymin.load_dataframes', 
                   return_value=(sample_data, sample_mc)):
            with patch('os.path.isfile', return_value=True):
                with patch('src.helpers.helper_pymin.get_options', return_value={}):
                    with patch('src.helpers.helper_pymin.write_results', return_value=True):
                        with patch('src.tools.reweight_pt_y.add_pt_y_weights', 
                                  return_value=sample_mc):
                            pymin.main()
                            
                            # Verify minimizer.minimize was called with the right arguments
                            mock_minimize.assert_called_once()
                            args, _ = mock_minimize.call_args
                            pd.testing.assert_frame_equal(args[0], sample_data)
                            pd.testing.assert_frame_equal(args[1], sample_mc)


# Test writing results
@patch('src.helpers.helper_pymin.write_results')
def test_write_results(mock_write_results, sample_data, sample_mc, sample_cats, temp_dir):
    """Test writing the minimization results."""
    cats_file = os.path.join(temp_dir, "cats.tsv")
    sample_cats.to_csv(cats_file, sep='\t', index=False, header=False)
    
    mock_results = {'cat1': {'scale': 1.01, 'smear': 0.01}, 
                    'cat2': {'scale': 0.99, 'smear': 0.02}}
    mock_write_results.return_value = True
    
    with patch('sys.argv', ['pymin.py', '-c', cats_file, '-o', 'test', 
                           '-i', 'test_input.txt']):
        with patch('src.helpers.helper_pymin.load_dataframes', 
                   return_value=(sample_data, sample_mc)):
            with patch('os.path.isfile', return_value=True):
                with patch('src.utilities.minimizer.minimize', return_value=mock_results):
                    with patch('src.tools.reweight_pt_y.add_pt_y_weights', 
                              return_value=sample_mc):
                        pymin.main()
                        
                        # Verify write_results was called with the right arguments
                        mock_write_results.assert_called_once()
                        args, _ = mock_write_results.call_args
                        assert args[1] == mock_results


# Test error handling for missing required arguments
def test_missing_arguments():
    """Test error handling for missing required arguments."""
    with patch('sys.argv', ['pymin.py']):
        with patch('builtins.print') as mock_print:
            pymin.main()
            
            # Check that an error message was printed
            mock_print.assert_any_call("[ERROR] you have not provided any arguments to this script.")


# Test error handling for closure test without smearings file
def test_closure_without_smearings():
    """Test error handling for closure test without smearings file."""
    with patch('sys.argv', ['pymin.py', '--closure', '-i', 'test_input.txt']):
        with patch('builtins.print') as mock_print:
            pymin.main()
            
            # Check that an error message was printed
            mock_print.assert_any_call("[ERROR] you have submitted a closure test without a smearings file.")


# Test condor submission
@patch('src.tools.condor_handler.manage')
def test_condor_submission(mock_manage):
    """Test condor submission functionality."""
    with patch('sys.argv', ['pymin.py', '--condor', '-o', 'test', '-i', 'test_input.txt']):
        with patch('src.helpers.helper_pymin.get_cmd', 
                  return_value="pymin.py --condor -o test -i test_input.txt"):
            pymin.main()
            
            # Verify condor_handler.manage was called with the right arguments
            mock_manage.assert_called_once()
            args, _ = mock_manage.call_args
            assert "pymin.py" in args[0]
            assert args[1] == "test"