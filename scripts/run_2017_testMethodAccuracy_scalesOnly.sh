./pymin.py \
    -i config/ul2017.dat \
    -c config/cats_step2.tsv \
    -o ul17_testMethodAccuracy_scaleOnly \
    --no-reweight \
    --test-method-accuracy \
    --closure \
    --condor \
    --queue='nextweek'
