./pymin.py \
    -i config/ul2018.dat \
    -c config/cats_step2.tsv \
    -o ul18_testMethodAccuracy_scaleOnly_v1 \
    --no-reweight \
    --scan-min=0.99 \
    --scan-max=1.01 \
    --scan-step=0.0005 \
    --min-step=0.0001 \
#    --closure \
#    --test-method-accuracy \
#    --condor \
#    --queue='nextweek'
