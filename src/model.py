import numpy as np
import pandas as pd

import os
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss, roc_curve, roc_auc_score, confusion_matrix
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from sklearn.neural_network import MLPClassifier

files = os.listdir('data/data_2016/') # dir is your directory path
file_set = set(files)
if '.ipynb_checkpoints' in file_set:
    files.remove('.ipynb_checkpoints')
    
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

agg_data['RIAGENDR'][agg_data['RIAGENDR']==2] = 0
agg_data['SMQ020'][agg_data['SMQ020'] > 1] = 0
agg_data['IND235'][(agg_data['IND235'] < 8) | (agg_data['IND235'] == 99) | (agg_data['IND235'] == 77) | (pd.isna(agg_data['IND235']))] = 0
agg_data['IND235'][agg_data['IND235'] >= 8] = 1
agg_data['DBD910'][agg_data['DBD910'] == 6666] = 91
agg_data['DBD910'][(agg_data['DBD910'] == 9999) | (pd.isna(agg_data['DBD910']))] = 2
agg_data['WHD140'][(agg_data['WHD140'] == 9999) | (agg_data['WHD140'] == 7777)] = 194
agg_data['BMXBMI'][pd.isna(agg_data['BMXBMI'])] = 29.4
agg_data['BPXPLS'][pd.isna(agg_data['BPXPLS'])] = 73
agg_data['MCQ300A'][(agg_data['MCQ300A'] > 1) | (pd.isna(agg_data['MCQ300A']))] = 0
limited_data = agg_data[['RIDAGEYR', 'RIAGENDR', 'BMXBMI', 'BPXPLS', 'DBD910', 'MCQ300A', 'IND235', 'SMQ020', 'WHD140']]

X_train, X_test, y_train, y_test = train_test_split(limited_data,cardio_risk)
scaler_train = MinMaxScaler()
scaler_test = MinMaxScaler()
X_train_scaled, X_test_scaled = scaler_train.fit_transform(X_train), scaler_test.fit_transform(X_test)

model_log = LogisticRegression(solver='lbfgs', max_iter=1000)
model_log.fit(X_train_scaled,y_train)
y_hat_log = model_log.predict_proba(X_test_scaled)
#log_loss(y_test, y_hat_log)