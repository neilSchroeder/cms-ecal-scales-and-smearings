./pymin.py -i config/ul16_post.dat \
    -c config/cats_step4_et.tsv \
    -s condor/ul16_postVFP_RunFineEtaR9Et_v00_2022-03-02/step4closure_ul16_postVFP_RunFineEtaR9Et_v00_2022-03-02_scales.dat \
    -w datFiles/ptz_x_rapidity_weights_ul16_postVFP_RunFineEtaR9_v00_2022-01-30.tsv \
    --smearings condor/ul16_postVFP_RunFineEtaR9_v00_2022-01-30/step3_ul16_postVFP_RunFineEtaR9_v00_2022-01-30_smearings.dat \
    -o ul16_postVFP_RunFineEtaR9Et_v01_2022-03-03 \
    --closure \
    --condor \
    --queue "nextweek"
