import numpy as np
import pandas as pd
import boto3
import io
import missingno as msno
from sklearn.impute import SimpleImputer, KNNImputer

s3 = boto3.client('s3')
obj = s3.get_object(Bucket='health-survey-tables', Key='2016/demo.csv')
table = pd.read_csv(io.BytesIO(obj['Body'].read())).set_index('SEQN')

key_list = []
s3objects = s3.list_objects_v2(Bucket='health-survey-tables', StartAfter='2016/' )
for object in s3objects['Contents']:
    key_list.append(object['Key'])
key_list.remove('2016/demo.csv')

for key in key_list:
    response = s3.get_object(Bucket='health-survey-tables', Key=key)
    table = pd.concat([table, pd.read_csv(io.BytesIO(response['Body'].read())).set_index('SEQN')],
                      axis=1,verify_integrity=True)

# create msno graphic showing missing data before cleaning
# msno.matrix(table)
# plt.savefig('img/missing_before.png')

#limiting data to adults who completed both survey & examination
adults = table[(table['RIDAGEYR']>=18) & (table['RIDSTATR']==2)]

#angina questions only asked of people age 40+
angina = adults[(adults['CDQ001'] == 1) & (adults['CDQ002'] == 1) & (adults['CDQ004'] == 1)
                 & (adults['CDQ005'] == 1) & (adults['CDQ006'] == 1)
                 & ((adults['CDQ009D'] == 4) | (adults['CDQ009E'] == 5))
                 | ((adults['CDQ009F'] == 6) & (adults['CDQ009G'] == 7))]

#history of CHD, angina, heart attack and stroke were taken for people age 20+
heart_history = adults[(adults['MCQ160C'] == 1) | (adults['MCQ160D'] == 1) | (adults['MCQ160E'] == 1)
                        | (adults['MCQ160F'] == 1)]

cardio_risk = pd.Series(np.zeros(adults.shape[0]), index=adults.index, name='cardio_risk')
for num in angina.index:
    cardio_risk[num] = 1
for num in heart_history.index:
    cardio_risk[num] = 1

#leaky data for cardio risk
adults.drop(adults.loc[:,'CDQ001':'CDQ010'], axis=1, inplace=True)
adults.drop(adults.loc[:,'MCQ160B':'MCQ180F'], axis=1, inplace=True)

#some featuring engineering is being done before related features are dropped
adults['DMDEDUC2'][((adults['RIDAGEYR']==18) | (adults['RIDAGEYR']==19)) & (adults['DMDEDUC3']==15)] = 1
adults['DMDEDUC2'][((adults['RIDAGEYR']==18) | (adults['RIDAGEYR']==19)) & (adults['DMDEDUC3']!=15)] = 0
sys = adults[['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']]
dia = adults[['BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']]
adults['AVGSYS'] = np.nanmean(sys, axis=1)
adults['AVGDIA'] = np.nanmean(dia, axis=1)
adults['DBD900'][(adults['DBD895']==0) &(pd.isna(adults['DBD900']))] = 0

#demo
adults.drop(['WTINT2YR', 'WTMEC2YR'], axis=1, inplace=True)
adults.drop(['SDDSRVYR', 'RIDSTATR', 'RIDAGEMN', 'RIDRETH1', 'DMQADFC', 'RIDEXMON', 'RIDEXAGM', 'DMDCITZN', 'DMDEDUC3'], axis=1, inplace=True)
adults.drop(adults.loc[:,'RIDEXPRG':'DMDHSEDU'], axis=1, inplace=True)
adults.drop(adults.loc[:,'SDMVPSU':'INDFMIN2'], axis=1, inplace=True)

#supps
adults.drop(['DSDANCNT', 'DSD010'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DSQTKCAL':'DSQTIODI'], axis=1, inplace=True)

#bp exam
adults.drop(adults.loc[:,'PEASCCT1':'BPACSZ'], axis=1, inplace=True)
adults.drop(adults.loc[:,'BPXPULS':'BPAEN4'], axis=1, inplace=True)

#body measure
adults.drop('BMDSADCM', axis=1, inplace=True)
adults.drop(adults.loc[:,'BMDSTATS':'BMIHT'], axis=1, inplace=True)
adults.drop(adults.loc[:,'BMDBMIC':'BMXSAD4'], axis=1, inplace=True)

#alcohol
adults.drop(adults.loc[:,'ALQ110':'ALQ130'], axis=1, inplace=True)
adults.drop(adults.loc[:,'ALQ141U':'ALQ160'], axis=1, inplace=True)

#bp questions
adults.drop(adults.loc[:,'BPD035':'BPQ050A'], axis=1, inplace=True)
adults.drop(adults.loc[:,'BPQ060':'BPQ100D'], axis=1, inplace=True)

#diet
adults.drop(adults.loc[:,'DBQ010':'DBQ073U'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DBQ223A':'DBQ424'], axis=1, inplace=True)
adults.drop(adults.loc[:,'CBQ596':'CBQ590'], axis=1, inplace=True)

#disability
adults.drop(adults.loc[:,'DLQ010':'DLQ080'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DLQ110':'DLQ170'], axis=1, inplace=True)

#insurance
adults.drop(adults.loc[:,'HIQ031A':'HIQ210'], axis=1, inplace=True)

#medical conditions
adults.drop(['MCQ160G', 'MCQ300B'], axis=1, inplace=True)
adults.drop(adults.loc[:,'MCQ010':'MCQ180N'], axis=1, inplace=True)
adults.drop(adults.loc[:,'MCQ180G':'MCQ240Z'], axis=1, inplace=True)
adults.drop(adults.loc[:,'MCQ370A':'OSQ230'], axis=1, inplace=True)

#occupation
adults.drop('OCD150', axis=1, inplace=True)
adults.drop(adults.loc[:,'OCQ210':'OCD395'], axis=1, inplace=True)

#physical activity
adults.drop(['PAQ610', 'PAD615', 'PAQ625', 'PAD630', 'PAQ640', 'PAD645', 'PAQ655', 'PAD660', 'PAQ670', 'PAD675',
             'PAD680', 'PAQ706'], axis=1, inplace=True)
adults.drop(adults.loc[:, 'PAQ722':'PAQ772C'], axis=1, inplace=True)

#sleep
adults.drop(['SLQ300', 'SLQ310', 'SLQ030', 'SLQ040', 'SLQ050'], axis=1, inplace=True)

#smoking
adults.drop(adults.loc[:,'SMQ050Q':'SMAQUEX2'], axis=1, inplace=True)

#weight history
adults.drop(['WHD110', 'WHD120', 'WHD130', 'WHQ190', 'WHQ200'], axis=1, inplace=True)
adults.drop(adults.loc[:,'WHD010':'WHD080L'], axis=1, inplace=True)

adults['RIAGENDR'][adults['RIAGENDR']==2] = 0
adults['DMQMILIZ'][adults['DMQMILIZ']==2] = 0
adults['DMDBORN4'][adults['DMDBORN4']>1] = 0
adults['BPQ020'][adults['BPQ020'] > 1] = 0
adults['BPQ080'][adults['BPQ080'] > 1] = 0
adults['DBQ700'][adults['DBQ700']==9] = 3
adults['DBQ197'][adults['DBQ197']==4] = np.array([3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 3, 2, 3, 3, 2, 2, 3, 2])
#adults['DBQ197'] hard-coded with result of np.random.randint
adults['DBD895'][adults['DBD895'] == 5555] = 22
adults['DBD895'][adults['DBD895'] == 9999] = round(adults['DBD895'][adults['DBD895'] != 9999].mean())
adults['HIQ011'][adults['HIQ011']>1] = 0
adults['MCQ365A'][adults['MCQ365A'] > 1] = 0
adults['MCQ365B'][adults['MCQ365B'] > 1] = 0
adults['MCQ365C'][adults['MCQ365C'] > 1] = 0
adults['MCQ365D'][adults['MCQ365D'] > 1] = 0
adults['PAQ605'][adults['PAQ605'] > 1] = 0
adults['PAQ620'][adults['PAQ620'] > 1] = 0
adults['PAQ635'][adults['PAQ635'] > 1] = 0
adults['PAQ650'][adults['PAQ650'] > 1] = 0
adults['PAQ665'][adults['PAQ665'] > 1] = 0
adults['PAQ710'][adults['PAQ710'] == 8] = 0
adults['PAQ710'][(adults['PAQ710'] == 99) | (adults['PAQ710'] == 77)] = round(adults['PAQ710'][(adults['PAQ710'] != 99) & (adults['PAQ710'] != 77)].mean())
adults['PAQ715'][adults['PAQ715'] == 8] = 0
adults['PAQ715'][(adults['PAQ715'] == 99) | (adults['PAQ715'] == 77)] = round(adults['PAQ715'][(adults['PAQ715'] != 99) & (adults['PAQ715'] != 77)].mean())
adults['SLQ120'][adults['SLQ120'] < 3] = 0
adults['SLQ120'][adults['SLQ120'] >= 3] = 1
adults['SMQ020'][adults['SMQ020'] > 1] = 0
adults['DSDCOUNT'][(adults['DSDCOUNT'] == 99) | (adults['DSDCOUNT'] == 77)] = round(adults['DSDCOUNT'][(adults['DSDCOUNT'] != 99) & (adults['DSDCOUNT'] != 77)].mean())
adults['DSD010AN'][adults['DSD010AN'] > 1] = 0
adults['WHQ225'][adults['WHQ225'] >= 5] = 0
adults['WHQ225'][adults['WHQ225'] < 5] = 1
adults['DMDYRSUS'][(adults['DMDYRSUS'] >= 4) | (pd.isna(adults['DMDYRSUS']))] = 1
adults['DMDYRSUS'][adults['DMDYRSUS'] < 4] = 0
adults['DMDMARTL'][(adults['DMDMARTL'] <= 4) | (adults['DMDMARTL'] == 77)] = 1
adults['DMDMARTL'][(adults['DMDMARTL'] > 4) | (pd.isna(adults['DMDMARTL']))] = 0
#adults['DMDMARTL'] missing values were all individuals under age 20, who were not asked the question
adults['DMDEDUC2'][(adults['DMDEDUC2'] <= 3) | (adults['DMDEDUC2'] == 9)] = 0
adults['DMDEDUC2'][adults['DMDEDUC2'] > 3] = 1
adults['ALQ101'][adults['ALQ101']==9] = np.nan
adults['ALQ141Q'][(adults['ALQ141Q'] == 777) | (adults['ALQ141Q'] == 999)] = np.nan
adults['ALQ141Q'][(adults['ALQ101']==2) & (pd.isna(adults['ALQ141Q']))] = 0
adults['BPQ030'][(adults['BPQ030'] > 1) | (pd.isna(adults['BPQ030']))] = 0
adults['DBD900'][(adults['DBD900'] == 9999) | (adults['DBD900'] == 7777)] = np.nan
adults['DBD900'][pd.isna(adults['DBD900'])] = round(np.nanmean(adults['DBD900']))
adults['DBD905'][adults['DBD905'] == 6666] = 91
adults['DBD905'][(adults['DBD905'] == 9999) | (adults['DBD905'] == 7777)] = np.nan
adults['DBD905'][pd.isna(adults['DBD905'])] = round(np.nanmean(adults['DBD905']))
adults['DBD910'][(adults['DBD910'] == 9999) | (adults['DBD910'] == 7777)] = np.nan
adults['DBD910'][pd.isna(adults['DBD910'])] = round(np.nanmean(adults['DBD910']))
adults['DLQ100'][(adults['DLQ100'] >= 3) | pd.isna(adults['DLQ100'])] = 0
adults['DLQ100'][adults['DLQ100'] < 3] = 1
adults['MCQ300A'][adults['MCQ300A'] == 2] = 0
adults['MCQ300A'][(adults['MCQ300A'] == 9) | (adults['MCQ300A'] == 7)] = np.nan
adults['MCQ300C'][adults['MCQ300C'] == 2] = 0
adults['MCQ300C'][(adults['MCQ300C'] == 9) | (adults['MCQ300C'] == 7)] = np.nan
adults['OCQ180'][(adults['OCQ180'] == 99999) | (adults['OCQ180'] == 77777)] = np.nan
adults['SLD012'][pd.isna(adults['SLD012'])] = round(np.nanmean(adults['SLD012']))
adults['SMD030'][pd.isna(adults['SMD030'])] = 0
adults['SMD030'][(adults['SMD030'] == 999) | (adults['SMD030'] == 777)] = round(adults['SMD030'][(adults['SMD030'] != 999) & (adults['SMD030'] != 777) & (adults['SMD030'] != 0)].mean())
adults['SMQ040'][adults['SMQ040'] == 2] = 1
adults['SMQ040'][(adults['SMQ040'] > 2) | (pd.isna(adults['SMQ040']))] = 0
adults['WHD140'][(adults['WHD140'] == 9999) | (adults['WHD140'] == 7777)] = np.nan
adults['WHQ150'][adults['WHQ150'] == 99999] = np.nan

race = pd.get_dummies(adults['RIDRETH3'], prefix='race')
diet_health = pd.get_dummies(adults['DBQ700'], prefix='diet_health')
milk = pd.get_dummies(adults['DBQ197'], prefix='milk')
data = pd.concat([adults, race, diet_health, milk], axis=1)
data.drop(['RIDRETH3', 'DBQ700', 'DBQ197'], axis=1, inplace=True)

# create msno graphic showing missing data after cleaning (before imputer)
# msno.matrix(data)
# plt.savefig('img/missing_after.png')

imputer = KNNImputer(n_neighbors=2, weights="uniform")
data[['MCQ300A', 'MCQ300C', 'ALQ101']] = imputer.fit_transform(data[['MCQ300A', 'MCQ300C', 'ALQ101']])
imputer2 = KNNImputer(n_neighbors=2, weights="uniform")
data[['ALQ141Q', 'INDFMPIR', 'BMXBMI', 'BMDAVSAD', 'BPXPLS', 'OCQ180', 'WHD140', 'WHQ150', 'AVGSYS', 'AVGDIA']] = imputer2.fit_transform(data[['ALQ141Q', 'INDFMPIR', 'BMXBMI', 'BMDAVSAD', 'BPXPLS', 'OCQ180', 'WHD140', 'WHQ150', 'AVGSYS', 'AVGDIA']])


'''
initial dropped features--
demo: 'SDDSRVYR', 'RIDSTATR', 'RIDAGEMN', 'RIDRETH1', 'DMQADFC', 'RIDEXMON', 'RIDEXAGM', 'DMDCITZN', 'DMDEDUC3', 'RIDEXPRG':'DMDHSEDU', 'SDMVPSU':'INDFMIN2'
nutrients1: 'DR1EXMER_nutrient1', 'DRABF_nutrient1', 'DR1DBIH_nutrient1':'DR1HELP_nutrient1', 'DR1STY':'DRQSDT91', 'DR1_330Z':'DR1TWS', 'DRD350A':'DRD350K', 'DRD370A':'DRD370V'
nutrients2: 'DR2EXMER_nutrient2', 'DRABF_nutrient2', 'DR2DBIH_nutrient2':'DR2HELP_nutrient2', 'DR2STY', 'DR2SKY', 'DR2_330Z':'DR2TWS'
supps: 'DSDANCNT', 'DSD010'
bp_exam: 'PEASCCT1':'BPACSZ', 'BPXPULS':'BPXML1', 'BPAEN1', 'BPAEN2', 'BPAEN3', 'BPAEN4'
body_measure: 'BMDSTATS':'BMIHT', 'BMDBMIC':'BMXSAD4', 'BMDSADCM'
alcohol: 'ALQ110':'ALQ130', 'ALQ141U':'ALQ160'
bp_questions: 'BPD035':'BPQ050A', 'BPQ060':'BPQ100D'
diet: 'DBQ010':'DBQ073U', 'DBQ229':'DBQ424', 'CBQ596':'CBQ590'
disability: 'DLQ010':'DLQ040', 'DLQ060', 'DLQ080', 'DLQ110', 'DLQ140':'DLQ170'
drugs: 'DUQ210':'DUQ215U', 'DUQ220Q':'DUQ240', 'DUQ260':'DUQ270U', 'DUQ280', 'DUQ300':'DUQ320', 'DUQ340':'DUQ350U', 'DUQ360':'DUQ430'
insurance: 'HIQ031A':'HIQ210'
medical_conditions: 'MCQ010':'MCQ053', 'MCD093':'MCQ180N', 'MCQ180G':'MCQ240Z', 'MCQ300B', 'MCQ370A':'OSQ230'
occupation: 'OCD150', 'OCQ210':'OCD395'
physical_activity: 'PAQ610', 'PAD615', 'PAQ625', 'PAD630', 'PAQ640', 'PAD645', 'PAQ655', 'PAD660', 'PAQ670', 'PAD675', 'PAD680', 'PAQ706', 'PAQ722':'PAQ772C'
sleep: 'SLQ300', 'SLQ310', 'SLQ030', 'SLQ040', 'SLQ050'
smoking: 'SMQ050Q':'SMD641', 'SMD093':'SMD100CO', 'SMD630':'SMQ852U', 'SMQ080':'SMAQUEX2'
weight_history: 'WHD010':'WHD080L', 'WHD130', 'WHQ190', 'WHQ200'

'''