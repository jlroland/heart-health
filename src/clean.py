import pandas as pd
import boto3
import io

s3 = boto3.client('s3')
obj = s3.get_object(Bucket='health-survey-tables', Key='2016/demo.csv')
data = pd.read_csv(io.BytesIO(obj['Body'].read())).set_index('SEQN')
key_list = []
s3objects = s3.list_objects_v2(Bucket='health-survey-tables', StartAfter='2016/' )
for object in s3objects['Contents']:
    key_list.append(object['Key'])
key_list.remove('2016/demo.csv')

for key in key_list:
    response = s3.get_object(Bucket='health-survey-tables', Key=key)
    data = pd.concat([data, pd.read_csv(io.BytesIO(response['Body'].read())).set_index('SEQN')], axis=1,verify_integrity=True)
adults = data[(data['RIDAGEYR']>=18) & (data['RIDSTATR']==2)]
weight_2yrint = adults['WTINT2YR']
weight_2yrmec = adults['WTMEC2YR']
adults.drop(['WTINT2YR', 'WTMEC2YR'], axis=1, inplace=True)

#cardio questions only asked of people age 40+
angina = adults[(adults['CDQ001'] == 1) & (adults['CDQ002'] == 1) & (adults['CDQ004'] == 1)
                 & (adults['CDQ005'] == 1) & (adults['CDQ006'] == 1)
                 & ((adults['CDQ009D'] == 4) | (adults['CDQ009E'] == 5))
                 | ((adults['CDQ009F'] == 6) & (adults['CDQ009G'] == 7))]

heart_history = adults[(adults['MCQ160C'] == 1) | (adults['MCQ160D'] == 1) | (adults['MCQ160E'] == 1)
                        | (adults['MCQ160F'] == 1)]

cardio_risk = pd.Series(np.zeros(adults.shape[0]), index=adults.index)
for num in angina.index:
    cardio_risk[num] = 1
for num in heart_history.index:
    cardio_risk[num] = 1

#leaky data for cardio risk
adults.drop(adults.loc[:,'CDQ001':'CDQ010'], axis=1, inplace=True)
adults.drop(adults.loc[:,'MCQ160B':'MCQ180F'], axis=1, inplace=True)

#demo
adults.drop(['SDDSRVYR', 'RIDSTATR', 'RIDAGEMN', 'RIDRETH1', 'RIDEXMON', 'RIDEXAGM', 'DMDCITZN', 'DMDEDUC3'], axis=1, inplace=True)
adults.drop(adults.loc[:,'RIDEXPRG':'DMDHSEDU'], axis=1, inplace=True)
adults.drop(adults.loc[:,'SDMVPSU':'INDFMIN2'], axis=1, inplace=True)

#nutrients
adults.drop(['DR1EXMER_nutrient1', 'DRABF_nutrient1'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DR1DBIH_nutrient1':'DR1HELP_nutrient1'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DR1STY':'DRQSDT91'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DR1_330Z':'DR1TWS'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DRD350A':'DRD350K'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DRD370A':'DRD370V'], axis=1, inplace=True)
adults.drop(['DR2EXMER_nutrient2', 'DRABF_nutrient2', 'DR2STY', 'DR2SKY'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DR2DBIH_nutrient2':'DR2HELP_nutrient2'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DR2_330Z':'DR2TWS'], axis=1, inplace=True)

#supps
adults.drop(['DSDANCNT', 'DSD010'], axis=1, inplace=True)

#bp exam
adults.drop(['BPAEN1', 'BPAEN2', 'BPAEN3', 'BPAEN4'], axis=1, inplace=True)
adults.drop(adults.loc[:,'PEASCCT1':'BPACSZ'], axis=1, inplace=True)
adults.drop(adults.loc[:,'BPXPULS':'BPXML1'], axis=1, inplace=True)

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
adults.drop(adults.loc[:,'DBQ229':'DBQ424'], axis=1, inplace=True)
adults.drop(adults.loc[:,'CBQ596':'CBQ590'], axis=1, inplace=True)

#disability
adults.drop(['DLQ060', 'DLQ080', 'DLQ110'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DLQ010':'DLQ040'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DLQ140':'DLQ170'], axis=1, inplace=True)

#drugs
adults.drop('DUQ280', axis=1, inplace=True)
adults.drop(adults.loc[:,'DUQ210':'DUQ215U'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DUQ220Q':'DUQ240'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DUQ260':'DUQ270U'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DUQ300':'DUQ320'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DUQ340':'DUQ350U'], axis=1, inplace=True)
adults.drop(adults.loc[:,'DUQ360':'DUQ430'], axis=1, inplace=True)

#insurance
adults.drop(adults.loc[:,'HIQ031A':'HIQ210'], axis=1, inplace=True)

#medical conditions
adults.drop('MCQ300B', axis=1, inplace=True)
adults.drop(adults.loc[:,'MCQ010':'MCQ053'], axis=1, inplace=True)
adults.drop(adults.loc[:,'MCD093':'MCQ180N'], axis=1, inplace=True)
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
adults.drop(adults.loc[:,'SMQ050Q':'SMD641'], axis=1, inplace=True)
adults.drop(adults.loc[:,'SMD093':'SMD100CO'], axis=1, inplace=True)
adults.drop(adults.loc[:,'SMD630':'SMQ852U'], axis=1, inplace=True)
adults.drop(adults.loc[:,'SMQ080':'SMAQUEX2'], axis=1, inplace=True)

#weight history
adults.drop(['WHD130', 'WHQ190', 'WHQ200'], axis=1, inplace=True)
adults.drop(adults.loc[:,'WHD010':'WHD080L'], axis=1, inplace=True)

'''
initial drop--
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