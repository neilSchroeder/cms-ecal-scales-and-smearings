#!/bin/bash
cd /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings
eval `scramv1 runtime -sh`  uname -a
echo $CMSSW_VERSION

./pymin.py  -i  config/ul16_pre.dat  -c  config/cats_step3_ul2016.tsv  -s  /afs/cern.ch/user/n/nschroed/public/scales/Run2016/Run2016_2021-02-01_UltraLegacy_preVFP_RunEtaR9_v2_scales.dat  -o  ul16_preVFP_RunFineEtaR9_v00_2022-01-28   --queue  nextweek  --from-condor

touch /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings/condor/ul16_preVFP_RunFineEtaR9_v00_2022-01-28/ul16_preVFP_RunFineEtaR9_v00_2022-01-28-done