./pymin.py -i config/ul16_post.dat \
    -c config/cats_step4_et.tsv \
    -s condor/ul16_postVFP_RunFineEtaR9Et_v03_2022-03-29/step-1_None_scales.dat \
    -w datFiles/ptz_x_rapidity_weights_ul16_postVFP_RunFineEtaR9_v00_2022-01-30.tsv \
    --smearings condor/ul16_postVFP_RunFineEtaR9_v00_2022-01-30/step3_ul16_postVFP_RunFineEtaR9_v00_2022-01-30_smearings.dat \
    -o ul16_postVFP_RunFineEtaR9Et_v04_2022-04-22 \
    --closure \
    --condor \
    --queue "nextweek"
