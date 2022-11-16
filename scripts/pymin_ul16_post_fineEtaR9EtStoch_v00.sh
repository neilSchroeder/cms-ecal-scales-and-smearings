./pymin.py -i config/ul16_post.dat \
    -c config/cats_step5_stochastic.tsv \
    -s condor/ul16_postVFP_RunFineEtaR9Et_v04_2022-04-22/step4closure_ul16_postVFP_RunFineEtaR9Et_v04_2022-04-22_scales.dat \
    -w datFiles/ptz_x_rapidity_weights_ul16_postVFP_RunFineEtaR9_v00_2022-01-30.tsv \
    -o ul16_postVFP_RunFineEtaR9EtStoch_v00_2022-04-25 \
    --condor \
    --queue "nextweek"
