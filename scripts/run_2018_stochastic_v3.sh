./pymin.py \
    -i config/ul2018.dat \
    -c config/cats_step2.tsv \
    -s ~/public/scales/Run2018/Run2018_2021-07-08_RunFineEtaR9Et_stochastic_v3-02_scales.dat \
    --smearings="/afs/cern.ch/user/n/nschroed/public/scales/Run2018/Run2018_2021-09-03_RunFineEtaR9Et_stochastic_v3-04_smearings.dat" \
    -w datFiles/ptz_x_rapidity_weights_ul18_2021-07-15_testOverhaul.tsv \
    -o ul18_2021-09-08_stochastic_v3-05 \
    --closure \
    --min-step=0.0001 \
    --scan-min=0.995 \
    --scan-max=1.005 \
    --scan-step=0.001 \
    --hist-min=84 \
    --hist-max=96 \
    --condor \
    --queue="tomorrow"