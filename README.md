# CMS - ECAL - Scales and Smearings

A new python framework for deriving the residual scales and additional smearings for the electron energy scale.

## Motivation

This project exists as a response to the state of the process of deriving the scales and smearings for the electron energy scale using ECALELF. 
The goal of this software is to improve usability, speed, and performance of the scales and smearings derivation. 
Additionally, this software serves as a portion of the thesis of Neil Schroeder from the University of Minnesota, Twin Cities School of Physics and Astronomy.

## Example Results

Here is an example of the kind of agreement that can be obtained between data and MC.
These results show UL17 data and MC with RunFineEtaR9Et scales and EtaR9Et smearings.

<img src="./examples/money_2017_ul_RunFineEtaR9Et_v3-01_EbEb_inclusive_wRatioPad.png" height="500" width="500">


## To Do

* Time permitting, or for whoever takes over development, multiprocessing the `zcat.update()` calls would likely speed things up.
* Implement `--systematics-study` feature in `pyval`. The idea is to automate estimating the systematic uncertainties.
* Change run divide feature to include/process fill infomation (more granular)


## Features

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

## Getting Started

Some basic instructions on how to get started:

### Prerequisites

The framework was built for use with python 3.6.4 on CMSSW_10_2_14.

### Installing

```
export SCRAM_ARCH=slc7_amd64_gcc700
cmsrel CMSSW_10_2_14
cd CMSSW_10_2_14/src/
git cms-init
git clone ssh://git@gitlab.cern.ch:7999/nschroed/cms-ecal-scales-and-smearings.git
cd cms-ecal-scales-and-smearings/
```
Now you'll want to checkout your own branch (name it something useful) and push it to the git repo
```
git branch myBranch
git push --set-upstream origin myBranch
git checkout myBranch
```

## Running the Framework

This framework has many options. To demonstrate it's uses, consider the following example:

### Ntuples

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
```

Note that the file size, compression, basket size, etc. are irrelevant. The requirement is the variable names.


### Basic 2018 Workflow

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
python3 pymin.py -i config/UltraLegacy2018.dat --prune -o 'pruned_ul18'
```
This takes your input files and will write them to tsvs in the folder DEST_PATH using the tage DEST_TAG

Now you will need to put the paths to the pruned files in a file, preferably in the config folder to run the run divider
```
python3 pymin.py -i config/ul2018.dat --run-divide -o ul18
```
If you want fewer run bins you can increase the default number of events per run using the `--minEvents` argument

With your run bins in hand you can now run the time_stability step:
```
python3 pymin.py -i config/ul2018.dat -c datFiles/run_divide_ul2018.dat -o ul2018 --time-stability
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
python3 pymin.py -i config/ul2018.dat -c config/cats_step2.tsv -s datFiles/step1_MY_TAG_scales.dat -o ul18_DATE_v0
```
This first step runs a derivation of both the scales and smearings

```
python3 pymin.py -i config/ul2018.dat \
           -c config/cats_step2.py \
           -s datFiles/step2_MY_TAG_scales.dat \
           -w datFiles/ptz_x_rapidity_weights_ul18_DATE_v0.tsv \
           -o ul18_step2_DATE_v0_closure \
           --smearings="datFiles/step2_ul18_DATE_v0_smearings.dat \
           --closure
```
This second step uses the `--closure` option and runs the minimization without any smearings. The MC is smeared ahead of the minimization using the smearings provided and no smearings are given to the minimizer. It can be useful to run this several times if your scales look off.

## Validation

Along side the `pymin.py` program comes the `pyval.py` program. This program is used to make validation plots which can be used to inspect the agreement of data and MC after application of the scales produced in `pymin.py`. 

### Setup

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

The basic usage looks like this:

```
python3 pyval.py \
    -i config/pyval/my_config.cfg \
    -o 'my_output_tag' \
    --data-title="Title Of Data" \
    --mc-title="Title Of MC" \
    --lumi-label="XX.X fb$^{-1}$ (13 TeV) 20XX" \
    --binning=NumBinsInHist \
    --write=/path/to/write/cleaned/events/
```

### Additional Options

pyval has the following additional options:

* `--log`: sets the logging level, this is mostly for debugging purposes
* `--systematics-study`: runs the systematics study by varying R9, Et, and working point ID. (Not yet working)

## Advanced Options and Additional Tools

What follows is a list of additional options that may be of some use as well as a list of tools which are helpful for the scales and smearings studies

### Advanced Options

To ignore specific categories by index use the `--ignore` option

To change the lower and upper bounds on the histograms used to evaluate the NLL of the dielectron categories use the `--hist-min` and `--hist-max` options

To turn off the auto-binning feature use the `--no-auto-bin` option.

To specify the bin size used in the NLL evaluation use the `--bin-size` option.

To change how the minimizer chooses the initial value of the scales and smearings use the `--start-style` option. The available choices are "scan", "random", and "specify".

To change the min and max values and the step size of the NLL scan used to seed the minimizer, use the `--scan-min`, `--scan-max`, and `--scan-step` options

To change the minimum step size the minimizer is allowed to take, use the `--min-step-size` option.

To fix the scales, and only derive a set of smearings, use the `--fix-scales` option. 

To submit the minimization to condor, use the `--condor` option, additionally you can specify the job flavour using the `--queue` options, the defualt queue is `tomorrow`

### Additional Use Options

To rewrite the scales/smearings file you've just created, rerun the same command with the `--rewrite` option

To merge an "only-step" file with a scales file, you can use the `--combine-files` option and provide the scales file with `-s` and the only step scales file with `--only-step`

### Plotting Options for pymin

To plot the 1D mass scans for each dielectron category provide the `--plot` option and provide the directory where the plots will be written with `--plot-dir`

### Advanced Diagnostic Options

To test the accuracy of the method, you can use the `--test-method-accuracy` option which will inject scales and smearings to MC in an attempt to derive the injected values back.

To scan the NLL phase space of a set of categories use the `--scan-nll` options, if you wish you specify the scales around which to scan you must also provide a config file to `--scan-scales`


### Additional Tools

This framework comes with some very useful tools that can be run independently from the minimizer or validator. Please see the python/tools/ page for more details

## Credit

Thanks to Shervin Nourbahksh, Peter Hansen, and Rajdeep Chatterjee for development on the previous scales and smearings code in ECALELF.
Thanks to Rajdeep Chatterjee for input on and review of this code.

## Contact

should something arise in which I must be contacted you can reach me at   
schr1077@umn.edu
