import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_curve, roc_auc_score, confusion_matrix

run src/clean.py        #cleans and imports data

#scatter plot of gender vs. age with class labels
jitter= 0.4 * np.random.random_sample(data['RIAGENDR'].shape) - 0.2
age_gender = data[['RIDAGEYR', 'RIAGENDR']]
colors = np.where(cardio_risk==1,'r','g')
fig,ax = plt.subplots(figsize=(10,8))
ax.scatter(x=data['RIDAGEYR'][cardio_risk==0],y=data['RIAGENDR'][cardio_risk==0] + jitter[:len(data['RIAGENDR'][cardio_risk==0])], alpha=0.2, c='green', 
           label='low-risk')
ax.scatter(x=data['RIDAGEYR'][cardio_risk==1],y=data['RIAGENDR'][cardio_risk==1] + jitter[-len(data['RIAGENDR'][cardio_risk==1]):], alpha=0.2, c='red', 
           label='high-risk')
ax.set_xlabel('Age (years)')
ax.set_yticks([0,1])
ax.set_yticklabels(['female', 'male'])
ax.set_ylabel('Gender')
ax.set_title('Risk Distribution by Age & Gender')
ax.legend()
#plt.savefig('img/age_gender_dist.png')

#Initial baseline model using age & gender
X_train, X_test, y_train, y_test = train_test_split(data,cardio_risk)

model_base = LogisticRegression(solver='lbfgs', max_iter=1000)
model_base.fit(X_train,y_train)
y_hat_base = model_base.predict_proba(X_test)
log_loss(y_test, y_hat_base), roc_auc_score(y_test, y_hat_base[:,1])

def get_ypred(y_true, y_pred, threshold=0.5):
    '''
    Takes output from predict_proba and returns class predictions at given threshold.
    '''
    y_thresh = [1 if y_pred[i,1] > threshold else 0 for i in range(len(y_pred))]
    return np.array(y_thresh)

#AUC score for baseline model seemed unusually high; should look at confusion matrix
cm = confusion_matrix(y_test, get_ypred(y_test, y_hat_base, 0.5))
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm, cmap='twilight')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1))
ax.yaxis.set(ticks=(0, 1))
ax.set_xticklabels(('Predicted 0', 'Predicted 1'), size='x-large')
ax.set_yticklabels(('Actual 0', 'Actual 1'), size='x-large')
ax.set_ylim(1.5, -0.5)
ax.set_title('Logistic, Threshold=0.5')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black', size='large')
#plt.savefig('cf_base_model.png')
plt.show()

scaler = MinMaxScaler()
X = scaler.fit_transform(data)

#find stable estimate of AUC using KFold iteration
kf = KFold(n_splits=5, shuffle=True, random_state=12)
auc_log = []
y_log = np.array(cardio_risk.copy())
for train, test in kf.split(X):
    model_test = LogisticRegression(solver='lbfgs', max_iter=1000)
    model_test.fit(X[train], y_log[train])
    y_hat_log = model_test.predict_proba(X[test])
    auc_log.append(roc_auc_score(y_log[test], y_hat_log[:,1]))
avg_auc_log = np.mean(auc_log)

auc_rf = []
y_rf = np.array(cardio_risk.copy())
for train, test in kf.split(X):
    model_test = RandomForestClassifier(n_estimators=1000, min_samples_split=10)
    model_test.fit(X[train], y_rf[train])
    y_hat_rf = model_test.predict_proba(X[test])
    auc_rf.append(roc_auc_score(y_rf[test], y_hat_rf[:,1]))
avg_auc_rf = np.mean(auc_rf)

auc_gbc = []
y_gbc = np.array(cardio_risk.copy())
for train, test in kf.split(X):
    model_test = GradientBoostingClassifier(n_estimators=1000, max_depth=2)
    model_test.fit(X[train], y_gbc[train])
    y_hat_gbc = model_test.predict_proba(X[test])
    auc_gbc.append(roc_auc_score(y_gbc[test], y_hat_gbc[:,1]))
avg_auc_gbc = np.mean(auc_gbc)

#Create ROC graphic comparing models
fpr_log, tpr_log, thresh_log = roc_curve(y_log[test], y_hat_log[:,1])
fpr_rf, tpr_rf, thresh_rf = roc_curve(y_rf[test], y_hat_rf[:,1])
fpr_gbc, tpr_gbc, thresh_gbc = roc_curve(y_gbc[test], y_hat_gbc[:,1])

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr_log, y=tpr_log,
                    mode='lines',
                    name=f'logistic, AUC={round(avg_auc_log, 2)}',
                    line=dict(color='red', width=3)))

fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf,
                    mode='lines',
                    name=f'random forest, AUC={round(avg_auc_rf, 2)}',
                    line=dict(color='green', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=fpr_gbc, y=tpr_gbc,
                    mode='lines',
                    name=f'gradient boosting, AUC={round(avg_auc_gbc, 2)}',
                    line=dict(color='blue', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                    mode='lines',
                    name='random',
                    line=dict(color='black', width=2)))

fig.update_layout(title='ROC Curves',
                   xaxis_title='fpr',
                   yaxis_title='tpr')

fig.show()
#fig.write_image('img/roc_comparison.png', width=1000, height=500)

#Confusion matrix for logistic regression model (best performance by small margin)
#Probability threshold = 0.5
cm = confusion_matrix(y_log[test], get_ypred(y_log[test], y_hat_log, 0.5))
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm, cmap='twilight')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1))
ax.yaxis.set(ticks=(0, 1))
ax.set_xticklabels(('Predicted 0', 'Predicted 1'), size='x-large')
ax.set_yticklabels(('Actual 0', 'Actual 1'), size='x-large')
ax.set_ylim(1.5, -0.5)
ax.set_title('Logistic, Threshold=0.5')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black', size='large')
plt.show()
#plt.savefig('img/cf_log.png')
