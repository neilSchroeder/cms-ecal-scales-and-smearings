#!/bin/bash
cd /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings
eval `scramv1 runtime -sh`  uname -a
echo $CMSSW_VERSION

./pymin.py  -i  config/ul2018.dat  -c  config/cats_step5_stochastic.tsv  -w  datFiles/ptz_x_rapidity_weights_ul18_2021-07-15_testOverhaul.tsv  -o  ul18_2021-07-15_testOverhaul_v3-04_testUpdatedScan   --queue=tomorrow  --from-condor

touch /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings/condor/ul18_2021-07-15_testOverhaul_v3-04_testUpdatedScan/ul18_2021-07-15_testOverhaul_v3-04_testUpdatedScan-done