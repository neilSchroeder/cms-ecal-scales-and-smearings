# CMS - ECAL - Scales and Smearings

A new python framework for deriving the residual scales and additional smearings for the electron energy scale.

## Getting Started

Some basic instructions on how to get started:

### Prerequisites

The framework was built for use with python 3.6.4 on CMSSW_10_2_14.

### Installing

```
cmsrel CMSSW_10_2_14
git clone https://gitlab.cern.ch/nschroed/cms-ecal-scales-and-smearings.git
```
Now you'll want to checkout your own branch (name it something useful)
```
git branch myBranch
```
And push it to your gitlab repo
```
git push --set-upstream origin myBranch
```

## Running the Framework

This framework has many options. To demonstrate it's uses, consider the following example:

### Basic 2018 Workflow

To start, you will need a file containing a list of data and mc files in the format 
```
type	treeName	filePath
```
where *type* is either "data" or "sim", *treeName* is the name of the tree in the root file containing the events you wish to analyze, and *filePath* is the full file path to the root file 

## Code Details

### optimize.py

## To Do
Add python/condor_helper.py to offer a default/automated method of submitting the job to condor
Add ignore feature to python/nll_wClass.py
