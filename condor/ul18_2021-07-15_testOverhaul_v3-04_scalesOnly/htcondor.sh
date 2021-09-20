#!/bin/bash
cd /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings
eval `scramv1 runtime -sh`  uname -a
echo $CMSSW_VERSION

./pymin.py  -i  config/ul2018.dat  -c  config/cats_step5_stochastic.tsv  -w  datFiles/ptz_x_rapidity_weights_ul18_2021-07-15_testOverhaul.tsv  -s  /afs/cern.ch/user/n/nschroed/public/scales/Run2018/Run2018_2021-07-08_RunFineEtaR9Et_stochastic_v3-02_scales.dat  --smearings=/afs/cern.ch/user/n/nschroed/public/scales/Run2018/Run2018_2021-07-06_RunFineEtaR9Et_stochastic_v3-01_smearings.dat  -o  ul18_2021-07-15_testOverhaul_v3-04_scalesOnly  --min-step=0.0001  --scan-min=0.995  --scan-max=1.005  --scan-step=0.001  --closure   --queue=nextweek  --from-condor

touch /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings/condor/ul18_2021-07-15_testOverhaul_v3-04_scalesOnly/ul18_2021-07-15_testOverhaul_v3-04_scalesOnly-done