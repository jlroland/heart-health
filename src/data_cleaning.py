import numpy as np
import pandas as pd
import os
from collections import defaultdict

#concatenate csv files in data folder into one dataframe
files = os.listdir('data/data_2016/')
file_set = set(files)
if '.ipynb_checkpoints' in file_set:
    files.remove('.ipynb_checkpoints')
    
number_files = len(files)

df_dict = defaultdict()
for file in files:
    df_dict[file.replace('.csv', '')] = pd.read_csv('data/data_2016/{}'.format(file)).set_index('SEQN')

agg_data = pd.concat(df_dict.values(), axis=1)
agg_data = agg_data[(agg_data['RIDAGEYR']>=18) & (agg_data['RIDSTATR']==2)]  #only considering adults


#create Series for target variable
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

#need to drop variables that were imported from csv files and won't be used in analysis
#demo data
agg_data.drop(['SDDSRVYR', 'RIDSTATR', 'RIDAGEMN', 'RIDRETH1', 'RIDEXMON', 'RIDEXAGM'], axis=1, inplace=True)
agg_data.drop(['DMQADFC', 'DMDCITZN', 'DMDEDUC3', 'DMDYRSUS'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'RIDEXPRG':'INDFMPIR'], axis=1, inplace=True)
agg_data = pd.concat([agg_data, pd.get_dummies(agg_data['RIDRETH3'], prefix='race')], axis=1)
agg_data.drop(['RIDRETH3', 'race_7.0'], axis=1, inplace=True)

#BP exam
agg_data.drop(['PEASCCT1', 'BPXCHR', 'BPAARM', 'BPACSZ', 'BPXPULS', 'BPXPTY', 'BPXML1'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:, 'BPAEN1':'BPAEN4'], axis=1, inplace=True)

#body measure
agg_data.drop(agg_data.loc[:, 'BMDSTATS':'BMIHT'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:, 'BMDBMIC':'BMIARMC'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:, 'BMXWAIST':'BMXSAD4'], axis=1, inplace=True)
agg_data.drop('BMDSADCM', axis=1, inplace=True)

#lab data
agg_data.drop('LBDTCSI', axis=1, inplace=True)

#diet behavior
agg_data.drop(agg_data.loc[:, 'DBQ010': 'DBQ424'], axis=1, inplace=True)
agg_data.drop('DBD900', axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'CBQ596': 'CBQ590'], axis=1, inplace=True)

#supps
agg_data.drop(['DSDANCNT', 'DSD010'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'DSQTKCAL': 'DSQTIODI'], axis=1, inplace=True)

#alcohol use
agg_data.drop(['ALQ110', 'ALQ120Q', 'ALQ120U', 'ALQ141U', 'ALQ151', 'ALQ160'], axis=1, inplace=True)

#smoking
agg_data.drop(['SMD030', 'SMQ040'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'SMQ050Q': 'SMAQUEX2'], axis=1, inplace=True)

#health insurance
agg_data.drop(agg_data.loc[:,'HIQ031A': 'HIQ210'], axis=1, inplace=True)

#income
agg_data.drop(agg_data.loc[:,'INQ020': 'INQ150'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:,'INDFMMPI':'INQ320'], axis=1, inplace=True)

#medical conditions
agg_data.drop(agg_data.loc[:, 'MCQ010':'MCQ053'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:, 'MCQ092':'MCQ180N'], axis=1, inplace=True)
agg_data.drop(['MCQ160G','MCQ180G', 'MCQ160K', 'MCQ160L'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:, 'MCQ170M':'MCQ240Z'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:, 'MCQ370A':'OSQ230'], axis=1, inplace=True)

#physical activity
agg_data.drop(['PAQ610', 'PAD615', 'PAQ625', 'PAD630', 'PAQ640', 'PAD645', 'PAQ655', 'PAD660', 'PAQ670', 'PAD675', 'PAD680',
               'PAQ706'], axis=1, inplace=True)
agg_data.drop(agg_data.loc[:, 'PAQ722':'PAQ772C'], axis=1, inplace=True)

#weight history
agg_data.drop(agg_data.loc[:, 'WHD010':'WHD130'], axis=1, inplace=True)
agg_data.drop(['WHQ190','WHQ200'], axis=1, inplace=True)

agg_data['RIAGENDR'][agg_data['RIAGENDR']==2] = 0
agg_data['DMQMILIZ'][agg_data['DMQMILIZ']==2] = 0
agg_data['DMDBORN4'][agg_data['DMDBORN4']>1] = 0
agg_data['HIQ011'][agg_data['HIQ011']>1] = 0
agg_data['SMQ020'][agg_data['SMQ020'] > 1] = 0
agg_data['DMDEDUC2'][(agg_data['DMDEDUC2'] <= 3) | (agg_data['DMDEDUC2'] == 9) | pd.isna(agg_data['DMDEDUC2'])] = 0
agg_data['DMDEDUC2'][agg_data['DMDEDUC2'] > 3] = 1
agg_data['DMDMARTL'][(agg_data['DMDMARTL'] <= 4) | (agg_data['DMDMARTL'] == 77) | pd.isna(agg_data['DMDMARTL'])] = 1
agg_data['DMDMARTL'][agg_data['DMDMARTL'] > 4] = 0
agg_data['DSDCOUNT'][(agg_data['DSDCOUNT'] == 99) | (agg_data['DSDCOUNT'] == 77)] = 1
agg_data['DSD010AN'][agg_data['DSD010AN'] > 1] = 0
agg_data['IND235'][(agg_data['IND235'] < 8) | (agg_data['IND235'] == 99) | (agg_data['IND235'] == 77) | (pd.isna(agg_data['IND235']))] = 0
agg_data['IND235'][agg_data['IND235'] >= 8] = 1
agg_data['DBD895'][agg_data['DBD895'] == 5555] = 22
agg_data['DBD895'][agg_data['DBD895'] == 9999] = 3
agg_data['DBD905'][(agg_data['DBD905'] == 9999) | (agg_data['DBD905'] == 7777) | (pd.isna(agg_data['DBD905']))] = 2
agg_data['DBD910'][agg_data['DBD910'] == 6666] = 91
agg_data['DBD910'][(agg_data['DBD910'] == 9999) | (pd.isna(agg_data['DBD910']))] = 2
agg_data['WHD140'][(agg_data['WHD140'] == 9999) | (agg_data['WHD140'] == 7777)] = 194
agg_data['WHQ150'][(agg_data['WHQ150'] == 99999) | (pd.isna(agg_data['WHQ150']))] = 40
agg_data['ALQ101'][(agg_data['ALQ101'] == 9) | (pd.isna(agg_data['ALQ101']))] = 1
agg_data['ALQ101'][agg_data['ALQ101'] == 2] = 0
agg_data['ALQ130'][agg_data['ALQ130'] == 999] = 3
agg_data['ALQ130'][pd.isna(agg_data['ALQ130'])] = 0
agg_data['ALQ141Q'][(agg_data['ALQ141Q'] == 777) | (agg_data['ALQ141Q'] == 999)] = 2
agg_data['ALQ141Q'][pd.isna(agg_data['ALQ141Q'])] = 0
agg_data['MCQ080'][agg_data['MCQ080'] > 1] = 0
agg_data['MCQ160M'][(agg_data['MCQ160M'] > 1) | (pd.isna(agg_data['MCQ160M']))] = 0
agg_data['MCQ300A'][(agg_data['MCQ300A'] > 1) | (pd.isna(agg_data['MCQ300A']))] = 0
agg_data['MCQ300B'][agg_data['MCQ300B'] > 1] = 0
agg_data['MCQ300C'][(agg_data['MCQ300C'] > 1) | (pd.isna(agg_data['MCQ300C']))] = 0
agg_data['MCQ365A'][agg_data['MCQ365A'] == 2] = 0
agg_data['MCQ365B'][agg_data['MCQ365B'] > 1] = 0
agg_data['MCQ365C'][agg_data['MCQ365C'] > 1] = 0
agg_data['MCQ365D'][agg_data['MCQ365D'] > 1] = 0
agg_data['PAQ605'][agg_data['PAQ605'] > 1] = 0
agg_data['PAQ620'][agg_data['PAQ620'] > 1] = 0
agg_data['PAQ635'][agg_data['PAQ635'] > 1] = 0
agg_data['PAQ650'][agg_data['PAQ650'] > 1] = 0
agg_data['PAQ665'][agg_data['PAQ665'] > 1] = 0
agg_data['PAQ710'][(agg_data['PAQ710'] == 99) | (agg_data['PAQ710'] == 77)] = 2
agg_data['PAQ710'][agg_data['PAQ710'] == 8] = 0
agg_data['PAQ710'][agg_data['PAQ710'] > 0] = 1
agg_data['PAQ715'][agg_data['PAQ715'] == 8] = 0
agg_data['PAQ715'][agg_data['PAQ715'] > 0] = 1
agg_data['BMXBMI'][pd.isna(agg_data['BMXBMI'])] = 29.4
agg_data['BMDAVSAD'][pd.isna(agg_data['BMDAVSAD'])] = 22.8
agg_data['BPXPLS'][pd.isna(agg_data['BPXPLS'])] = 73
agg_data['BPXSY1'][pd.isna(agg_data['BPXSY1'])] = 125
agg_data['BPXDI1'][pd.isna(agg_data['BPXDI1'])] = 70
agg_data['LBXTC'][pd.isna(agg_data['LBXTC'])] = 189

code_list = list(agg_data.columns)
description_list = ['income > $45K/year', 'num meals not prepared at home', 'num RTE foods', 'num frozen meals',
                   'pulse', 'systolic pressure', 'diastolic pressure', 'total cholesterol', 
                    'smoked 100+ cigs in life', 'do vigorous work', 'do moderate work', 'walk/bike regularly',
                   'vigorous recreation', 'moderate recreation', '1+ hrs/day TV', 
                    '1+ hrs/day computer/video games', 'most ever weighed', 'age when heaviest', 
                   'had 12+ drinks in past year', 'avg num drinks/day', 'num days binge drink in past year',
                   'doctor ever say overweight', 'doctor ever say thyroid problem', 'close relative heart attack',
                   'close relative asthma', 'close relative diabetes', 'doctor say lose weight', 'doctor say exercise',
                   'doctor say reduce salt', 'doctor say reduce fat/calories', 'BMI', 'avg SAD', 'gender', 'age',
                   'served in military', 'born in U.S.', 'education beyond HS', 'ever married', 'num supplements taken',
                   'taken antacids', 'covered by health insurance', 'Mexican American', 'Other Hispanic',
                   'Non-Hispanic White', 'Non-Hispanic Black', 'Non-Hispanic Asian']
features = {code: description for code, description in zip(code_list,description_list)}
agg_data.rename(mapper=features, axis='columns', inplace=True)

#back-up of column name dictionary

# {'IND235': 'income > $45K/year',
#  'DBD895': 'num meals not prepared at home',
#  'DBD905': 'num RTE foods',
#  'DBD910': 'num frozen meals',
#  'BPXPLS': 'pulse',
#  'BPXSY1': 'systolic pressure',
#  'BPXDI1': 'diastolic pressure',
#  'LBXTC': 'total cholesterol',
#  'SMQ020': 'smoked 100+ cigs in life',
#  'PAQ605': 'do vigorous work',
#  'PAQ620': 'do moderate work',
#  'PAQ635': 'walk/bike regularly',
#  'PAQ650': 'vigorous recreation',
#  'PAQ665': 'moderate recreation',
#  'PAQ710': '1+ hrs/day TV',
#  'PAQ715': '1+ hrs/day computer/video games',
#  'WHD140': 'most ever weighed',
#  'WHQ150': 'age when heaviest',
#  'ALQ101': 'had 12+ drinks in past year',
#  'ALQ130': 'avg num drinks/day',
#  'ALQ141Q': 'num days binge drink in past year',
#  'MCQ080': 'doctor ever say overweight',
#  'MCQ160M': 'doctor ever say thyroid problem',
#  'MCQ300A': 'close relative heart attack',
#  'MCQ300B': 'close relative asthma',
#  'MCQ300C': 'close relative diabetes',
#  'MCQ365A': 'doctor say lose weight',
#  'MCQ365B': 'doctor say exercise',
#  'MCQ365C': 'doctor say reduce salt',
#  'MCQ365D': 'doctor say reduce fat/calories',
#  'BMXBMI': 'BMI',
#  'BMDAVSAD': 'avg SAD',
#  'RIAGENDR': 'gender',
#  'RIDAGEYR': 'age',
#  'DMQMILIZ': 'served in military',
#  'DMDBORN4': 'born in U.S.',
#  'DMDEDUC2': 'education beyond HS',
#  'DMDMARTL': 'ever married',
#  'DSDCOUNT': 'num supplements taken',
#  'DSD010AN': 'taken antacids',
#  'HIQ011': 'covered by health insurance',
#  'race_1.0': 'Mexican American',
#  'race_2.0': 'Other Hispanic',
#  'race_3.0': 'Non-Hispanic White',
#  'race_4.0': 'Non-Hispanic Black',
#  'race_6.0': 'Non-Hispanic Asian'}