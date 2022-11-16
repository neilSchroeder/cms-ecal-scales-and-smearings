./pymin.py -i config/ul16_pre.dat \
    -c config/cats_step3_ul2016.tsv \
    -s condor/ul16_preVFP_RunFineEtaR9_v00_2022-01-30/step3_ul16_preVFP_RunFineEtaR9_v00_2022-01-30_scales.dat \
    -w datFiles/ptz_x_rapidity_weights_ul16_preVFP_RunFineEtaR9_v00_2022-01-30.tsv \
    --smearings condor/ul16_preVFP_RunFineEtaR9_v00_2022-01-30/step3_ul16_preVFP_RunFineEtaR9_v00_2022-01-30_smearings.dat \
    -o ul16_preVFP_RunFineEtaR9_v01_2022-02-28 \
    --closure \
    --condor \
    --queue "nextweek"
