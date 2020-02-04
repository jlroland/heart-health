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

#leaky data for cardio risk
agg_data.drop(agg_data.loc[:,'CDQ001':'CDQ010'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'MCQ160B':'MCQ180F'], axis=1, inplace=True)


#demo data
agg_data.drop(['SDDSRVYR', 'RIDSTATR', 'RIDAGEMN', 'RIDRETH1', 'RIDEXMON', 'RIDEXAGM'], axis=1, inplace=True)
agg_data.drop(['DMQADFC', 'DMDCITZN', 'DMDEDUC3'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'RIDEXPRG':'INDFMPIR'], axis=1, inplace=True)

#BP exam
agg_data.drop(['PEASCCT1', 'BPXCHR', 'BPAARM', 'BPACSZ', 'BPXPULS', 'BPXPTY', 'BPXML1', 'BPAEN1', 'BPAEN2', 'BPAEN3', 'BPAEN4'], axis=1, inplace=True)

#body measure
agg_data.drop(agg_data.loc[:, 'BMDSTATS':'BMIHT'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:, 'BMDBMIC':'BMIARMC'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:, 'BMXWAIST':'BMXSAD4'], axis=1, inplace=True)
agg_data.drop('BMDSADCM', axis=1, inplace=True)

#lab data
agg_data.drop(['WTSAF2YR','LBDGLUSI', 'LBDHDDSI', 'LBDTRSI', 'LBDLDLSI', 'LBDHRPLC', 'LBDINSI',
               'LBDINLC'], axis=1, inplace=True)

#diet behavior
agg_data.drop(agg_data.loc[:, 'DBQ010': 'DBQ424'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'CBQ596': 'CBQ590'], axis=1, inplace=True)

#supps
agg_data.drop(['DSDANCNT', 'DSD010'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'DSQTKCAL': 'DSQTIODI'], axis=1, inplace=True)

#alcohol use
agg_data.drop(['ALQ110', 'ALQ120Q', 'ALQ120U', 'ALQ141U', 'ALQ151', 'ALQ160'], axis=1, inplace=True)

#drug use
agg_data.drop(agg_data.loc[:,'DUQ210': 'DUQ240'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'DUQ260': 'DU280'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'DUQ300': 'DU320'], axis=1, inplace=True)
agg_data.drop('DUQ280', axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'DUQ340': 'DUQ430'], axis=1, inplace=True)


#health insurance
agg_data.drop(agg_data.loc[:,'HIQ031A': 'HIQ210'], axis=1, inplace=True)

#income
agg_data.drop(agg_data.loc[:,'INQ020': 'INQ150'], axis=1, inplace=True)
agg_data.drop('INQ320', axis=1, inplace=True)

#medical conditions
agg_data.drop(agg_data.loc[:, 'MCQ010':'MCQ053'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:, 'MCQ092':'MCQ180N'], axis=1, inplace=True)
agg_data.drop(['MCQ180G', 'MCQ170M', 'MCQ180M', 'MCQ170K', 'MCQ180K', 'MCQ170L', 'MCQ180L'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:, 'MCQ203':'MCQ240Z'], axis=1, inplace=True)
agg_data.drop('OSQ230', axis=1, inplace=True)

#physical activity
agg_data.drop(agg_data.loc[:, 'PAQ722':'PAQ772C'], axis=1, inplace=True)