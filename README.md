# CMS - ECAL - Scales and Smearings

A new python framework for deriving the residual scales and additional smearings for the electron energy scale.

## Table of Contents
- [Motivation](#motivation)
- [Example Results](#example-results)
- [To Do](#to-do)
- [Features](#features)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installing](#installing)
        - [Installing Anaconda](#installing-anaconda)
    - [Running the Framework](#running-the-framework)
        - [Ntuples](#ntuples)
        - [Basic 2018 Workflow](#basic-2018-workflow)
- [Validation](#validation)
    - [Setup](#setup)
    - [Usage](#usage)
        - [Additional Options](#additional-options)
- [Advanced Options and Additional Tools](#advanced-options-and-additional-tools)
    - [Advanced Options](#advanced-options)
    - [Additional Use Options](#additional-use-options)
    - [Plotting Options for pymin](#plotting-options-for-pymin)
    - [Advanced Diagnostic Options](#advanced-diagnostic-options)
    - [Additional Tools](#additional-tools)
- [Modifying the Framework](#modifying-the-framework)
    - [New Variables](#new-variables)
    - [New Minimization Strategy](#new-minimization-strategy)
    - [New Loss Function](#new-loss-function)
    - [New Z Categories](#new-z-categories)
    - [New Plot Style](#new-plot-style)
    - [New Plot Title](#new-plot-title)
- [Credit](#credit)
- [Contact](#contact)

## Motivation

[table of contents](#table-of-contents)

This project exists as a response to the state of the process of deriving the scales and smearings for the electron energy scale using ECALELF. 
The goal of this software is to improve usability, speed, and performance of the scales and smearings derivation. 
Additionally, this software serves as a portion of the thesis of Neil Schroeder from the University of Minnesota, Twin Cities School of Physics and Astronomy.

## Example Results

[table of contents](#table-of-contents)


Here is an example of the kind of agreement that can be obtained between data and MC.
These results show UL17 data and MC with RunFineEtaR9Et scales and EtaR9Et smearings.

<img src="./examples/money_2017_ul_RunFineEtaR9Et_v3-01_EbEb_inclusive_wRatioPad.png" height="500" width="500">


## To Do

[table of contents](#table-of-contents)


* Time permitting, or for whoever takes over development, multiprocessing the `zcat.update()` calls would likely speed things up.
* I think there's a better way to parallelize the application of the scales.
* Properly implement `--systematics-study` feature in `pyval`. The idea is to automate estimating the systematic uncertainties.
* Change run divide feature to include/process lumisection infomation (more granular).

Potential improvements (for a thesis or summer project):
* Run divide: include lumi information
* Scanning: 
    * 1D scans from input scales
    * Track loss during minimization
* Loss function exploration
    * Dynamic learning rate
    * Smearings loss function: can we make it steeper near the minimum

## Features

[table of contents](#table-of-contents)


This software has a number of interesting features:
* A pruner to convert root files into tsv files with only relevant branches
* A run divider to derive run bins 
* A time stabilizer which uses medians to stabilize the scale as a function of run number
* A minimizer to evaluate the scales and smearings:
	* Auto-binning of dielectron category invariant mass distributions using the Freedman-Diaconis Rule
    * Numba histograms to dramatically increase the speed of binning invariant mass distributions and NLL evaluation
    * 1D scanning or random start styles of the scales/smearings for the minimizer
    * SciPi minimizer using the 'L-BFGS-B' method for speed and memory preservation
* A program for producing plots showing the agreement of data and MC
    * Any variable from the trees can be plotted
    * Cuts on both leading and subleading Eta, Et, and R9 can be made
    * different styles of plots can be selected for different kinds of validation
* A program for testing the accuracy of the method by injecting scales and smearings into MC and attempting to derive them back
* A program for evaluating the systematic uncertainties of the scales+smearings by varying R9, Et, and working point ID

## Getting Started

[table of contents](#table-of-contents)


Some basic instructions on how to get started:

### Prerequisites

[table of contents](#table-of-contents)


The framework was built for use with python 3.6.4 on CMSSW_10_2_14.

### Installing

[table of contents](#table-of-contents)


Due to some concerns over changing environments on lxplus, the new installation instructions use Anaconda

#### Installing Anaconda

First, check if you have access to `conda` by running:
`which conda`
if you get an output that looks like this (on lxplus):
```
conda ()
{
    \local cmd="${1-__missing__}";
    case "$cmd" in
        activate | deactivate)
            __conda_activate "$@"
        ;;
        install | update | upgrade | remove | uninstall)
            __conda_exe "$@" || \return;
            __conda_reactivate
        ;;
        *)
            __conda_exe "$@"
        ;;
    esac
}
```
or like this (on your local machine):
```
/home/<user>/anaconda3/bin/conda
```
then you already have access to anaconda and you shouldn't need to do a new installation.


otherwise you can install anaconda3 like so:
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
```
When installing Anaconda3 you'll probably want to install it in your work repo because it's large and trying to put it into your basic lxplus box will limit your space significantly. It will prompt you to pick an install location, enter: `/afs/cern.ch/work/<u>/<user>/anaconda3` replacing `<u>` and `<user>` with the first initial of your username and your username, respectively. This takes a while, be patient.

Once you've installed anaconda you'll be asked to reboot your shell. 

With anaconda installed, navigate to a location you'd like to install this repo (your /afs/cern.ch/work/ is recommended).

```
cd <target-directory>
git clone ssh://git@gitlab.cern.ch:7999/nschroed/cms-ecal-scales-and-smearings.git
cd cms-ecal-scales-and-smearings
conda env create -f env.yml
conda activate scales-env
```

Now you'll want to checkout your own branch (name it something useful) and push it to the git repo
```
git branch myBranch
git push --set-upstream origin myBranch
git checkout myBranch
```

## Running the Framework

[table of contents](#table-of-contents)


This framework has many options. To demonstrate it's uses, consider the following example:

### Ntuples

[table of contents](#table-of-contents)


You will need a set of root files for data and simulation that must have a tree named `selected` with the following branches:

```
******************************************************************************
*Tree    :selected  : selected                                               *
******************************************************************************
*Br    0 :runNumber : runNumber/i                                            *
*............................................................................*
*Br    1 :R9Ele     : R9Ele[3]/F                                             *
*............................................................................*
*Br    2 :etaEle    : etaEle[3]/F                                            *
*............................................................................*
*Br    3 :phiEle    : phiEle[3]/F                                            *
*............................................................................*
*Br    4 :energy_ECAL_ele : energy_ECAL_ele[3]/F                             *
*............................................................................*
*Br    5 :invMass_ECAL_ele : invMass_ECAL_ele/F                              *
*............................................................................*
*Br    6 :gainSeedSC : gainSeedSC[3]/b                                       *
*............................................................................*
*Br    7 :eleID     : eleID[3]/i                                             *
*............................................................................*
```

Note that the file size, compression, basket size, etc. are irrelevant. The requirement is the variable names.


### Basic 2018 Workflow

[table of contents](#table-of-contents)


To start, you will need a file containing a list of data and mc files in the format 
```
type	treeName	filePath
```
for example, from `config/UltraLegacy2018.dat`:
```
data    selected    /eos/cms/store/group/dpg_ecal/alca_ecalcalib/ecalelf/ntuples/13TeV/MINIAODNTUPLE/106X_dataRun2_UL18/EGamma-106X-2018UL-T1-A/315257-316995/2018_314472-325175_SS/2018UL_SS/EGamma-106X-2018UL-T1-A-315257-316995.root
```

where *type* is either "data" or "sim", *treeName* is the name of the tree in the root file containing the events you wish to analyze, and *filePath* is the full file path to the root file. An example of this can be seen in config/UltraLegacy2018.dat

You can now run the pruner:
```
python pymin.py -i config/UltraLegacy2018.dat --prune -o 'pruned_ul18'
```
This takes your input files and will write them to tsvs in the folder DEST_PATH using the tage DEST_TAG

Now you will need to put the paths to the pruned files in a file, preferably in the config folder to run the run divider
```
python pymin.py -i config/pruned_ul18.cfg --run-divide -o ul18
```
If you want fewer run bins you can increase the default number of events per run using the `--minEvents` argument

With your run bins in hand you can now run the time_stability step:
```
python pymin.py -i config/pruned_ul18.cfg -c datFiles/run_divide_ul18.dat -o ul18 --time-stability
```

You can also run the following instead, if you want to plot the results of the time stabilization:
```
python pymin.py -i config/pruned_ul18.cfg -c datFiles/run_divide_ul18.dat -o ul18 --time-stability --plot --lumi-label '59.7 fb^{-1} (13 TeV) 2018'
```
From here you can run the scales and smearings chain. This requires a couple additional ingredients.
The first is a categories file, you can see an example below:
```
#type	etaMin	etaMax	r9Min	r9Max	gain	etMin	etMax
scale	0.	1.	0.	0.96	-1	-1	-1
scale	1.	1.4442	0.	0.96	-1	-1	-1
scale	1.566	2.	0.	0.96	-1	-1	-1
scale	2.	2.5	0.	0.96	-1	-1	-1
scale	0.	1.	0.96	10.	-1	-1	-1
scale	1.	1.4442	0.96	10.	-1	-1	-1
scale	1.566	2.	0.96	10.	-1	-1	-1
scale	2.	2.5	0.96	10.	-1	-1	-1
smear	0.	1.	0.	0.96	-1	-1	-1
smear	1.	1.4442	0.	0.96	-1	-1	-1
smear	1.566	2.	0.	0.96	-1	-1	-1
smear	2.	2.5	0.	0.96	-1	-1	-1
smear	0.	1.	0.96	10.	-1	-1	-1
smear	1.	1.4442	0.96	10.	-1	-1	-1
smear	1.566	2.	0.96	10.	-1	-1	-1
smear	2.	2.5	0.96	10.	-1	-1	-1
```

The categories are defined for single electrons, and di-electron categories are built during the minimization process.  
Please be extra careful when building your categories to ensure that you do not skip coverage in a variable.

Step2 is coarseEtaR9, step3 is fineEtaR9, step4 is either fineEtaR9Gain, or fineEtaR9Et:

```
python pymin.py -i config/pruned_ul18.cfg -c config/cats_step2.tsv -s datFiles/step1_ul18_scales.dat -o ul18
```
This first step runs a derivation of both the scales and smearings

```
python pymin.py -i config/pruned_ul18.cfg \
           -c config/cats_step2.py \
           -s datFiles/step2_MY_TAG_scales.dat \
           -w datFiles/ptz_x_rapidity_weights_ul18_DATE_v0.tsv \
           -o ul18_step2_DATE_v0_closure \
           --smearings="datFiles/step2_ul18_DATE_v0_smearings.dat \
           --closure
```
This second step uses the `--closure` option and runs the minimization without any smearings. The MC is smeared ahead of the minimization using the smearings provided and no smearings are given to the minimizer. It can be useful to run this several times if your scales look off.

Because the minimizer can take a long time to derive the scales and smearings, there is a built-in option to submit the script as a job to condor. To do this simply run the above commands appending `--condor --queue <queue>` like so:

```
python pymin.py -i config/pruned_ul18.cfg -c config/cats_step2.tsv -s datFiles/step1_ul18_scales.dat -o ul18 --condor --queue <queue>
```
and 
```
python pymin.py -i config/pruned_ul18.cfg \
           -c config/cats_step2.py \
           -s datFiles/step2_MY_TAG_scales.dat \
           -w datFiles/ptz_x_rapidity_weights_ul18_DATE_v0.tsv \
           -o ul18_step2_DATE_v0_closure \
           --smearings="datFiles/step2_ul18_DATE_v0_smearings.dat \
           --closure \
           --condor --queue <queue>
```


## Validation

[table of contents](#table-of-contents)


Along side the `pymin.py` program comes the `pyval.py` program. This program is used to make validation plots which can be used to inspect the agreement of data and MC after application of the scales produced in `pymin.py`. 

### Setup

[table of contents](#table-of-contents)


To get started you'll need a .cfg file to provide to `pyval`. The .cfg file contains tab separated values and is structured as follows:

```
DATA    path/to/data/csv/file.csv
MC  path/to/MC/csv/file.csv
SCALES  path/to/scales/file.dat
SMEARINGS   path/to/smearings/file.dat
WEIGHTS path/to/pt/and/rapidity/weights/for/mc.tsv
CATS    path/to/category/definition/file.tsv
```

The data, mc, scales, smearings, and weights files are all produced by `pymin` and should already exist, but the category definition file is one you'll either have to make, or adjust the example files available to you.

The category definition file is a .tsv file and is structured as follows:

```
style   name    variable    eta0    r90 et0    eta1    r91 et1
plotStyle   nameOfPlot  variableToPlot  (minLeadEta,maxLeadEta) (minLeadR9,maxLeadR9)   (minLeadEt,maxLeadEt)   (minSubEta,maxSubEta)   (minSubR9,maxSubR9) (minSubEt,maxSubEt)
```

For most plots, you'll choose the style `paper`, and the variable will likely be `invMass_ECAL_ele`. 
If you don't want to place a cut on a particular variable, just set the min and max to -1 like so: `(-1,-1)`.

An example of the category definition file can be found in `config/pyval/plot_cats_standard.tsv`.

### Usage

[table of contents](#table-of-contents)


The basic usage looks like this:

```
python pyval.py \
    -i config/pyval/my_config.cfg \
    -o 'my_output_tag' \
    --data-title="Title Of Data" \
    --mc-title="Title Of MC" \
    --lumi-label="XX.X fb$^{-1}$ (13 TeV) 20XX" \
    --binning=NumBinsInHist \
    --write=/path/to/write/cleaned/events/
```

### Additional Options

[table of contents](#table-of-contents)


pyval has the following additional options:

* `--log`: sets the logging level, this is mostly for debugging purposes
* `--systematics-study`: runs the systematics study by varying R9, Et, and working point ID. (Not yet working)

## Advanced Options and Additional Tools

[table of contents](#table-of-contents)


What follows is a list of additional options that may be of some use as well as a list of tools which are helpful for the scales and smearings studies

### Advanced Options

[table of contents](#table-of-contents)


To run the minimization or the plotting with a smaller dataset to debug parts of the code, use the `--debug` option. This will run the code on only 1000000 events.

To ignore specific categories by index use the `--ignore` option

To change the lower and upper bounds on the histograms used to evaluate the NLL of the dielectron categories use the `--hist-min` and `--hist-max` options

To turn off the auto-binning feature use the `--no-auto-bin` option.

To specify the bin size used in the NLL evaluation use the `--bin-size` option.

To change how the minimizer chooses the initial value of the scales and smearings use the `--start-style` option. The available choices are "scan", "random", and "specify".

To change the min and max values and the step size of the NLL scan used to seed the minimizer, use the `--scan-min`, `--scan-max`, and `--scan-step` options

To change the minimum step size the minimizer is allowed to take, use the `--min-step-size` option.

To fix the scales, and only derive a set of smearings, use the `--fix-scales` option. 

To submit the minimization to condor, use the `--condor` option, additionally you can specify the job flavour using the `--queue` options, the defualt queue is `tomorrow`

To refine a specific set of scales or smearings, use the `--start-style` option paired with the `--only-step` option and a file in the `only_step` format with the desired scales and smearings. like so:

```
python pymin.py -i /path/to/data.cfg \
           -c /path/to/categories.tsv \
           -s /path/to/scales.dat \
           -w /path/to/weights.tsv \
           -o <output_tag> \
           --start-style 'specify' \
           --only-step /path/to/only/step.tsv
           --condor --queue <queue>
```

Note: only step files have the following format:
```
runmin	runmax	etamin	etamax	r9min	r9max	etmin	etmax	gain	scaleOrSmear	unc
```
for the purposes of use in `--start-style` the binning is for your benefit. The program will read in the values without checking the binning. Please ensure the order of your binning matches the order of the binning in the categories file.

### Additional Use Options

[table of contents](#table-of-contents)


To rewrite the scales/smearings file you've just created, rerun the same command with the `--rewrite` option

To merge an "only-step" file with a scales file, you can use the `--combine-files` option and provide the scales file with `-s` and the only step scales file with `--only-step`

### Plotting Options for pymin

[table of contents](#table-of-contents)


To plot the 1D mass scans for each dielectron category provide the `--plot` option and provide the directory where the plots will be written with `--plot-dir`

### Advanced Diagnostic Options

[table of contents](#table-of-contents)


To test the accuracy of the method, you can use the `--test-method-accuracy` option which will inject scales and smearings to MC in an attempt to derive the injected values back.

To scan the NLL phase space of a set of categories use the `--scan-nll` options, if you wish you specify the scales around which to scan you must also provide a config file to `--scan-scales`


### Additional Tools

[table of contents](#table-of-contents)


This framework comes with some very useful tools that can be run independently from the minimizer or validator. Please see the python/tools/ page for more details

## Modifying the Framework

[table of contents](#table-of-contents)


There are numerous reasons to need to modify the network. I'll try to list the ones I anticipate to be the most common, from practical experience, here.

### New Variables

[table of contents](#table-of-contents)


If you want to add new variables to the CSV file dumped in the pruning step, you can check the [keep](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/classes/constant_classes.py?ref_type=heads#L68) and [drop](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/classes/constant_classes.py?ref_type=heads#L69) lists in python/classes/constant_classes.py and add the variable you want to put in the CSV. 

Then if you want to apply cuts to that variable, you'll want to update the [python/utilities/data_loader.py](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/utilities/data_loader.py) to add cut options and a function for this variable.

If this variable is going to be used in minimization or category definitions you'll also need to edit either the [minimizer](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/utilities/minimizer.py), [helper_minimizer](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/helpers/helper_minimizer.py), [zcat_class](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/classes/zcat_class.py), or [data_loader.py/extract_cats](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/utilities/data_loader.py) to incorporate the new variable into the minimization.

### New Minimization Strategy

[table of contents](#table-of-contents)


If you want to change the minimization strategy you can change that in [constant_classes.py](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/classes/constant_classes.py?ref_type=heads#L92) and you can find the documentation for [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html), the recommended approaches are "L-BFGS-B" or "Nelder-Mead".

If you want to completely overhaul the minimization engine you can do so by schanging [minimize.py](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/utilities/minimizer.py?ref_type=heads#L218) accordingly.

### New Loss Function

[table of contents](#table-of-contents)


If you'd like to re-work the loss function you can do so in [minimize.py](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/helpers/helper_minimizer.py?ref_type=heads#L218) and [zcat_class.py]()

### New Z Categories

[table of contents](#table-of-contents)


### New Plots

[table of contents](#table-of-contents)

#### New Plot Style

[table of contents](#table-of-contents)

To add a new style of plot, for the sake of example NewStyle, you will need to do the following:
- create a new style file under python/plotters/plot_styles. If the style is called "New" then make a file called `new_style.py`. You may want to use `paper_style.py` as a template. It is the most detailed example.
- add your plotting function to [plots.py]() with a name like `plot_style_new()`, do not forget to import the new style at the beginning of the file. 
- In [make_plots.py]()
    - import the style
    - add the style as a new entry to `pc.plotting_functions`
    


#### New Plot Title

[table of contents](#table-of-contents)

Simply add a new entry to the 'TITLE_ANOTTATION' dictionary in [plot_style.py](https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings/-/blob/master/python/plotters/plot_styles/plot_style.py). The key should be the name you provide in your pyval config file, the title must be a string. If you want to use special LaTeX characters you must use a [raw string](https://realpython.com/python-raw-strings/).


## Credit

[table of contents](#table-of-contents)


Thanks to Shervin Nourbahksh, Peter Hansen, and Rajdeep Chatterjee for development on the previous scales and smearings code in ECALELF.
Thanks to Rajdeep Chatterjee and Onuray Sancar for input on and review of this code.

## Contact

[table of contents](#table-of-contents)


should something arise in which I must be contacted you can reach me at   
schr1077@umn.edu
