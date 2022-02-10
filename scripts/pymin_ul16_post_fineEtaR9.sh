./pymin.py -i config/ul16_post.dat \
    -c config/cats_step3_ul2016.tsv \
    -s "/afs/cern.ch/user/n/nschroed/public/scales/Run2016/Run2016_2021-02-01_UltraLegacy_postVFP_RunEtaR9_v3_scales.dat" \
    -o ul16_postVFP_RunFineEtaR9_v00_2022-01-30 \
    --condor \
    --queue "nextweek"
