import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix

run src/clean.py

scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(data,cardio_risk, random_state=42)
X = pd.concat([X_train, y_train], axis=1)
low_risk = X[X.cardio_risk==0]
high_risk = X[X.cardio_risk==1]

# upsample minority
high_upsampled = resample(high_risk,
                         replace=True, # sample with replacement
                         n_samples=len(low_risk), # match number in majority class
                         random_state=42) # reproducible results

upsampled = pd.concat([low_risk, high_upsampled])
y_train_upsampled = upsampled['cardio_risk']
X_train_upsampled = upsampled.drop('cardio_risk', axis=1)
X_train_upsampled_scaled, X_test_scaled = scaler.fit_transform(X_train_upsampled), scaler.transform(X_test)

model_log = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
model_log.fit(X_train_upsampled_scaled, y_train_upsampled)
y_hat_log = model_log.predict_proba(X_test_scaled)
log_loss(y_test, y_hat_log), roc_auc_score(y_test, y_hat_log[:,1])

model_rf = RandomForestClassifier(n_estimators=1000, min_samples_split=10)
model_rf.fit(X_train_upsampled, y_train_upsampled)
y_hat_rf = model_rf.predict_proba(X_test)
log_loss(y_test, y_hat_rf), roc_auc_score(y_test, y_hat_rf[:,1])

model_gbc = GradientBoostingClassifier(n_estimators=1000, max_depth=2)
model_gbc.fit(X_train_upsampled, y_train_upsampled)
y_hat_gbc = model_gbc.predict_proba(X_test)
log_loss(y_test, y_hat_gbc), roc_auc_score(y_test, y_hat_gbc[:,1])

def calculate_recall(cm):
    '''
    Parameters:
    cm (array): 2x2 confusion matrix
    Returns:
    float: calculated recall based on tp and fn values in matrix
    '''
    return cm[1,1]/sum(cm[1,:])

def get_ypred(y_true, y_pred, threshold=0.5):
    '''
    Takes output from predict_proba and returns class predictions at given threshold.
    '''
    y_thresh = [1 if y_pred[i,1] > threshold else 0 for i in range(len(y_pred))]
    return np.array(y_thresh)

cm_log = confusion_matrix(y_test, get_ypred(y_test, y_hat_log, 0.5))
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm_log, cmap='twilight')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1))
ax.yaxis.set(ticks=(0, 1))
ax.set_xticklabels(('Predicted 0', 'Predicted 1'), size='x-large')
ax.set_yticklabels(('Actual 0', 'Actual 1'), size='x-large')
ax.set_ylim(1.5, -0.5)
ax.set_title('Logistic, Threshold=0.5')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_log[i, j], ha='center', va='center', color='black', size='large')
fig.tight_layout()
#plt.savefig('img/cf_log_upsample.png')
plt.show()

calculate_recall(cm_log)

cm_rf = confusion_matrix(y_test, get_ypred(y_test, y_hat_rf, 0.5))
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm_rf, cmap='twilight')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1))
ax.yaxis.set(ticks=(0, 1))
ax.set_xticklabels(('Predicted 0', 'Predicted 1'), size='x-large')
ax.set_yticklabels(('Actual 0', 'Actual 1'), size='x-large')
ax.set_ylim(1.5, -0.5)
ax.set_title('Random Forest, Threshold=0.5')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_rf[i, j], ha='center', va='center', color='black', size='large')
fig.tight_layout()
#plt.savefig('img/cf_rf_upsample.png')
plt.show()

calculate_recall(cm_rf)

cm_gbc = confusion_matrix(y_test, get_ypred(y_test, y_hat_gbc, 0.5))
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm_gbc, cmap='twilight')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1))
ax.yaxis.set(ticks=(0, 1))
ax.set_xticklabels(('Predicted 0', 'Predicted 1'), size='x-large')
ax.set_yticklabels(('Actual 0', 'Actual 1'), size='x-large')
ax.set_ylim(1.5, -0.5)
ax.set_title('Gradient Boost, Threshold=0.5')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_gbc[i, j], ha='center', va='center', color='black', size='large')
fig.tight_layout()
#plt.savefig('img/cf_gbc_upsample.png')
plt.show()

calculate_recall(cm_gbc)

log_features = pd.DataFrame({'Fraction of Samples Affected' : model_log.coef_[0]}, 
                            index=data.columns)
log_features = log_features.sort_values(by='Fraction of Samples Affected')



limited_features = data[['INDFMPIR', 'BPXPLS', 'SMD030', 'race_1.0', 'BMXBMI', 'DSDCOUNT', 'DBD895', 'race_6.0', 'race_4.0', 'race_3.0', 'race_2.0', 'milk_0.0', 'RIAGENDR', 'race_7.0', 'SMQ020', 'MCQ365C', 'BPQ020', 'PAQ710', 'MCQ300A', 'WHD140', 'RIDAGEYR']]

features = {'INDFMPIR':'income/poverty ratio', 'BPXPLS':'pulse', 'SMD030':'age started smoking regularly', 'race_1.0':'Mexican American', 'BMXBMI':'bmi', 'DSDCOUNT':'num dietary supplements taken', 'DBD895':'num meals not home-prepared', 'AVGDIA':'avg diastolic pressure', 'race_6.0':'Non-Hispanic Asian', 'HIQ011':'has health insurance', 'SMQ020':'smoked 100+ cigarettes in life', 'MCQ365C':'told to reduce salt', 'BPQ020':'told had high bp', 'PAQ710':'num hours/day watching TV/videos', 'MCQ300A':'close relative had heart attack', 'WHD140':'most ever weighed', 'RIDAGEYR':'age'}

log_features_reduced = log_features[np.absolute(log_features['Fraction of Samples Affected']) >= 0.4]
log_features_reduced = log_features_reduced.rename(index=features)
fig, ax = plt.subplots(figsize=(6,8))
ax.barh(log_features_reduced.index, log_features_reduced['Fraction of Samples Affected'].values, height=0.8)
ax.set_xlabel('Beta Coefficients')
ax.set_title('Logistic Regression Feature Weights')
fig.tight_layout()
#plt.savefig('img/feature_importance_reduced.png')

X_train_limited, X_test_limited, y_train_limited, y_test_limited = train_test_split(limited_features,cardio_risk, stratify=cardio_risk, random_state=22)
X_limited = pd.concat([X_train_limited, y_train_limited], axis=1)

low_risk_limited = X_limited[X_limited.cardio_risk==0]
high_risk_limited = X_limited[X_limited.cardio_risk==1]

# upsample minority
high_upsampled_limited = resample(high_risk_limited,
                         replace=True, # sample with replacement
                         n_samples=len(low_risk_limited), # match number in majority class
                         random_state=42) # reproducible results


upsampled_limited = pd.concat([low_risk_limited, high_upsampled_limited])
y_train_upsampled_limited = upsampled_limited['cardio_risk']
X_train_upsampled_limited = upsampled_limited.drop('cardio_risk', axis=1)

model_limited = Pipeline([
    ('scale', MinMaxScaler()),
    ('log', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000))
    ])
model_limited.fit(X_train_upsampled_limited, y_train_upsampled_limited)
y_hat_limited = model_limited.predict_proba(X_test_limited)
log_loss(y_test_limited, y_hat_limited), roc_auc_score(y_test_limited, y_hat_limited[:,1])

cm_limited = confusion_matrix(y_test_limited, get_ypred(y_test_limited, y_hat_limited, 0.5))
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(cm_limited, cmap='twilight')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1))
ax.yaxis.set(ticks=(0, 1))
ax.set_xticklabels(('Predicted Low', 'Predicted High'), size='x-large')
ax.set_yticklabels(('Actual Low', 'Actual High'), size='x-large')
ax.set_ylim(1.5, -0.5)
ax.set_title('Logistic (Limited), Threshold=0.5')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_limited[i, j], ha='center', va='center', color='black', size='large')
fig.tight_layout()
#plt.savefig('img/cf_log_upsample_limited.png')
plt.show()

calculate_recall(cm_limited)

# pickle_file = open('model.pickle', 'wb')
# pickle.dump(model_limited, pickle_file)
# pickle_file.close()
