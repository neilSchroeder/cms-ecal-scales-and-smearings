#!/bin/bash
cd /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings
eval `scramv1 runtime -sh`  uname -a
echo $CMSSW_VERSION

./pymin.py  -i  config/ul16_pre.dat  -c  config/cats_step3_ul2016.tsv  -s  condor/ul16_preVFP_RunFineEtaR9_v01_2022-01-19/step3closure_ul16_preVFP_RunFineEtaR9_v01_2022-01-19_scales.dat  -o  ul16_preVFP_RunFineEtaR9_v02_2022-01-24   --queue  nextweek  --from-condor

touch /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings/condor/ul16_preVFP_RunFineEtaR9_v02_2022-01-24/ul16_preVFP_RunFineEtaR9_v02_2022-01-24-done