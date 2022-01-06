#!/bin/bash
cd /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings
eval `scramv1 runtime -sh`  uname -a
echo $CMSSW_VERSION

./pymin.py  -i  config/ul16_post.dat  -c  config/cats_step3_ul2016.tsv  -s  /afs/cern.ch/user/n/nschroed/public/scales/Run2016/Run2016_2021-02-01_UltraLegacy_postVFP_RunEtaR9_v3_scales.dat  -o  ul16_postVFP_step3_2021-12-22_v0  --smearings  /afs/cern.ch/user/n/nschroed/public/scales/Run2016/Run2016_2021-01-29_UltraLegacy_postVFP_RunEtaR9_v0_smearings.dat  --closure   --queue  nextweek  --from-condor

touch /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings/condor/ul16_postVFP_step3_2021-12-22_v0/ul16_postVFP_step3_2021-12-22_v0-done