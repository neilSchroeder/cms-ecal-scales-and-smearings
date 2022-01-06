./pymin.py -i config/ul16_pre.dat \
    -c config/cats_step3_ul2016.tsv \
    -s ~/public/scales/Run2016/Run2016_2021-02-01_UltraLegacy_preVFP_RunEtaR9_v2_scales.dat \
    -o ul16_preVFP_step3_2021-12-22_v0 \
    --smearings "/afs/cern.ch/user/n/nschroed/public/scales/Run2016/Run2016_2021-02-04_UltraLegacy_preVFP_RunFineEtaR9_v0_smearings.dat" \
    --closure \
    --condor \
    --queue "nextweek"
