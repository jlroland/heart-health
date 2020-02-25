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
adults = data[data['RIDAGEYR'] >= 18]


'''
demo: 'SDDSRVYR', 'RIDAGEMN', 'RIDRETH1', 'RIDEXMON', 'RIDEXAGM', 'DMDCITZN', 'DMDEDUC3', 'RIDEXPRG':'DMDHSEDU', 'SDMVPSU':'INDFMIN2'
nutrients1: 'DR1EXMER_nutrient1', 'DRABF_nutrient1', 'DR1DBIH_nutrient1':'DR1HELP_nutrient1', 'DR1STY':'DRQSDT91', 'DR1_330Z':'DR1TWS', 'DRD350A':'DRD350K', 'DRD370A':'DRD370V'
nutrients2: 'DR2EXMER_nutrient2', 'DRABF_nutrient2', 'DR2DBIH_nutrient2':'DR2HELP_nutrient2', 'DR2STY', 'DR2SKY', 'DR2_330Z':'DR2TWS'
supps: 'DSDANCNT', 'DSD010'
bp_exam: 'PEASCCT1':'BPACSZ', 'BPXPULS':'BPXML1', 'BPAEN1', 'BPAEN2', 'BPAEN3', 'BPAEN4'
body_measure: 'BMDSTATS':'BMIHT', 'BMDBMIC':'BMXSAD4', 'BMDSADCM'
alcohol: 'ALQ110':'ALQ130', 'ALQ141U':'ALQ160'
bp_questions: 'BPD035':'BPQ050A', 'BPQ060':'BPQ100D'


'''