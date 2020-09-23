import pandas as pd

step4 = pd.read_csv("datFiles/onlystep4_2017_et_scales_v13a.dat",delimiter='\t',header=None)
step5 = pd.read_csv("datFiles/onlystep4_2017_et_scales.dat",delimiter='\t',header=None)

print(step5)

for i,row5 in step5.iterrows():
    for j,row4 in step4.iterrows():
        if row5[2] <= row4[2] and row5[3] >= row4[3]:
            if row5[6] <= row4[6] and row5[7] >= row4[7]:
                step4.iloc[j,9] *= step5.iloc[i,9]

step4.to_csv("datFiles/onlystep4_2017_et_scales_v13b.dat", sep='\t', index=False, header=False)

