./pymin.py -i config/ul16_pre.dat \
    -c config/cats_step2.tsv \
    -s condor/ul16_preVFP_RunFineEtaR9Et_v00_2022-03-03/step4closure_ul16_preVFP_RunFineEtaR9Et_v00_2022-03-03_scales.dat \
    -w datFiles/ptz_x_rapidity_weights_ul16_preVFP_RunFineEtaR9_v00_2022-01-30.tsv \
    --smearings condor/ul16_preVFP_RunFineEtaR9_v00_2022-01-30/step3_ul16_preVFP_RunFineEtaR9_v00_2022-01-30_smearings.dat \
    -o ul16_preVFP_RunFineEtaR9Et_v01_2022-03-29 \
    --closure \
    --condor \
    --queue "nextweek"
