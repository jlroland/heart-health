import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import os
from collections import defaultdict

files = os.listdir('data/data_2016/') # dir is your directory path
number_files = len(files)

df_dict = defaultdict()
for file in files:
    df_dict[file.replace('.csv', '')] = pd.read_csv('data/data_2016/{}'.format(file)).set_index('SEQN')

agg_data = pd.concat(df_dict.values(), axis=1)
agg_data = agg_data[(agg_data['RIDAGEYR']>=18) & (agg_data['RIDSTATR']==2)]  #only considering adults

#demo data
# agg_data.drop(['SDDSRVYR', 'RIDSTATR', 'RIDAGEMN', 'RIDRETH1', 'RIDEXMON', 'RIDEXAGM'], axis=1, inplace=True)
# agg_data.drop(agg_data.loc[:,'RIDEXPRG':'AIALANGA'], axis=1, inplace=True)
# agg_data.drop(agg_data.loc[:,'DMDHRGND':'INDFMPIR'], axis=1, inplace=True)

#BP exam
#agg_data.drop(['PEASCCT1', 'BPAARM', 'BPACSZ', 'BPXPULS', 'BPXPTY', 'BPXML1', 'BPAEN1', 'BPAEN2', 'BPAEN3', 'BPAEN4'], axis=1, inplace=True)

#body measure
#agg_data.drop(['BMIWT', 'BMIRECUM', 'BMXHEAD', 'BMIHEAD', 'BMIHT', 'BMXLEG', 'BMILEG', 'BMXARML', 'BMXARMC', 
               'BMIARMC', 'BMIWAIST', 'BMXSAD1', 'BMXSAD2', 'BMXSAD3', 'BMXSAD4', 'BMDSADCM'], axis=1, inplace=True)

#lab data
# agg_data.drop(['LBDGLUSI', 'LBDHDDSI', 'LBDTRSI', 'LBDLDLSI', 'LBDHRPLC', 'LBDINSI',
#                'LBDINLC'], axis=1, inplace=True)
# agg_data.drop(['LBDGLTSI', 'GTDSCMMN', 'GTDDR1MN', 'GTDBL2MN', 'GTDDR2MN', 'GTXDRANK', 
#                'GTDCODE'], axis=1, inplace=True)

#diet behavior
# agg_data.drop(agg_data.loc[:, 'DBQ010': 'DBQ424'], axis=1, inplace=True)
# agg_data.drop(agg_data.loc[:,'CBQ596': 'CBQ590'], axis=1, inplace=True)

#health insurance
#agg_data.drop(agg_data.loc[:,'HIQ031A': 'HIQ210'], axis=1, inplace=True)

#income
# agg_data.drop(agg_data.loc[:,'INQ020': 'INQ150'], axis=1, inplace=True)
# agg_data.drop(['INQ320'], axis=1, inplace=True)

#medical conditions
# agg_data.drop(agg_data.loc[:, 'MCQ149':'MCQ180N'], axis=1, inplace=True)
# agg_data.drop(agg_data.loc[:, 'MCQ220':'MCQ240Z'], axis=1, inplace=True)
# agg_data.drop('OSQ230', axis=1, inplace=True)

#physical activity
#agg_data.drop(agg_data.loc[:, 'PAQ722':'PAQ772C'], axis=1, inplace=True)

#cardio questions only asked of people age 40+
angina = agg_data[(agg_data['CDQ001'] == 1) & (agg_data['CDQ002'] == 1) & (agg_data['CDQ004'] == 1)
                 & (agg_data['CDQ005'] == 1) & (agg_data['CDQ006'] == 1)
                 & ((agg_data['CDQ009D'] == 4) | (agg_data['CDQ009E'] == 5))
                 | ((['CDQ009F'] == 6) & (['CDQ009G'] == 7))]

heart_history = agg_data[(agg_data['MCQ160C'] == 1) | (agg_data['MCQ160D'] == 1) | (agg_data['MCQ160E'] == 1)
                        | (agg_data['MCQ160F'] == 1)]

cardio_risk = pd.Series(np.zeros(agg_data.shape[0]), index=agg_data.index)
for num in angina.index:
    cardio_risk[num] = 1
for num in heart_history.index:
    cardio_risk[num] = 1

#