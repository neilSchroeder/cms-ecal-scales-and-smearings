./pymin.py -i config/ul16_post.dat \
    -c config/cats_step2_eb.tsv \
    -s condor/ul16_postVFP_RunFineEtaR9Et_v02_2022-03-21/step4closure_ul16_postVFP_RunFineEtaR9Et_v02_2022-03-21_scales.dat \
    -w datFiles/ptz_x_rapidity_weights_ul16_postVFP_RunFineEtaR9_v00_2022-01-30.tsv \
    --smearings condor/ul16_postVFP_RunFineEtaR9_v00_2022-01-30/step3_ul16_postVFP_RunFineEtaR9_v00_2022-01-30_smearings.dat \
    -o ul16_postVFP_RunFineEtaR9Et_v03_2022-03-29 \
    --closure \
    --condor \
    --queue "nextweek"
