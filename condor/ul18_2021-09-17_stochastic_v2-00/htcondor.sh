#!/bin/bash
cd /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings
eval `scramv1 runtime -sh`  uname -a
echo $CMSSW_VERSION

./pymin.py  -i  config/ul2018.dat  -c  config/cats_step4_et.tsv  -s  /afs/cern.ch/user/n/nschroed/public/scales/Run2018/step2closure_ul18_2021-09-08_stochastic_v3-05_scales.dat  --smearings=/afs/cern.ch/user/n/nschroed/public/scales/Run2018/Run2018_2021-09-03_RunFineEtaR9Et_stochastic_v3-05_smearings.dat  -w  datFiles/ptz_x_rapidity_weights_ul18_2021-07-15_testOverhaul.tsv  -o  ul18_2021-09-17_stochastic_v2-00  --closure  --min-step=0.0001  --scan-min=0.995  --scan-max=1.005  --scan-step=0.001  --hist-min=82  --hist-max=98   --queue=nextweek  --from-condor

touch /afs/cern.ch/work/n/nschroed/ss_pyfit/CMSSW_10_2_14/src/cms-ecal-scales-and-smearings/condor/ul18_2021-09-17_stochastic_v2-00/ul18_2021-09-17_stochastic_v2-00-done