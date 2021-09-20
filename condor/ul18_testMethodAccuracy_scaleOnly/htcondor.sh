#!/bin/bash
cd /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings
eval `scramv1 runtime -sh`  uname -a
echo $CMSSW_VERSION

./pymin.py  -i  config/ul2018.dat  -c  config/cats_step2.tsv  -o  ul18_testMethodAccuracy_scaleOnly  --no-reweight  --scan-min=0.99  --scan-max=1.01  --scan-step=0.0005  --min-step=0.0001  --test-method-accuracy  --closure   --queue=nextweek  --from-condor

touch /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings/condor/ul18_testMethodAccuracy_scaleOnly/ul18_testMethodAccuracy_scaleOnly-done