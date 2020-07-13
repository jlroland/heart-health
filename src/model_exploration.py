import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss, roc_curve, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

run src/clean.py

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
X_train_init, X_test_init, y_train_init, y_test_init = train_test_split(agg_data[['age', 'gender']],cardio_risk)
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_init,y_train_init)
y_prob_init = model.predict_proba(X_test_init)
#log_loss(y_test_init, y_prob_init)
roc_auc_score(y_test_init, y_prob_init[:,1])

#do train-test-split for data, with & without normalization
X_train, X_test, y_train, y_test = train_test_split(agg_data,cardio_risk)
scaler_train = MinMaxScaler()
scaler_test = MinMaxScaler()
X_train_scaled, X_test_scaled = scaler_train.fit_transform(X_train), scaler_test.fit_transform(X_test)

#Logistic Regression with feature scaling and regularization
model_log = LogisticRegression(solver='lbfgs', max_iter=10000)
model_log.fit(X_train_scaled,y_train)
y_hat_log = model_log.predict_proba(X_test_scaled)
log_loss(y_test, y_hat_log)

#try using only top 10 features by importance (based on coefficients)
# ind = list(np.absolute(model.coef_).argsort()[0])
# top10_log = agg_data.iloc[:,ind[-10:]]
# X_train10_log, X_test10_log, y_train10_log, y_test10_log = train_test_split(top10_log, cardio_risk)
# model10_log = LogisticRegression(C=1000, solver='lbfgs', max_iter=10000)
# model10_log.fit(X_train10_log, y_train10_log)
# y_hat10_log = model10_log.predict_proba(X_test10_log)
# log_loss(y_test10_log, y_hat10_log)

model_rf = RandomForestClassifier(n_estimators=1000, min_samples_split=10)
model_rf.fit(X_train, y_train)
y_hat_rf = model_rf.predict_proba(X_test)
log_loss(y_test, y_hat_rf)

model_gbc = GradientBoostingClassifier(n_estimators=1000, max_depth=2)
model_gbc.fit(X_train, y_train)
y_hat_gbc = model_gbc.predict_proba(X_test)
log_loss(y_test, y_hat_gbc)

neural = MLPClassifier(hidden_layer_sizes=(100, 2), activation='logistic',alpha=0.0001, max_iter=200)
neural.fit(X_train, y_train)
y_hat_neural = neural.predict_proba(X_test)
log_loss(y_test, y_hat_neural)

roc_auc_score(y_test, y_hat_log[:,1])  #logistic regression
roc_auc_score(y_test, y_hat_rf[:,1]) #random forest
roc_auc_score(y_test, y_hat_gbc[:,1])    #gradient boosting classifier
roc_auc_score(y_test, y_hat_neural[:,1])  #neural network

#find stable estimate of ROC AUC using KFold iteration
kf = KFold(n_splits=10, shuffle=True)
auc = []
scaler_k = MinMaxScaler()
X = scaler_k.fit_transform(agg_data)
y = cardio_risk
for train, test in kf.split(X):
    model_test = LogisticRegression(solver='lbfgs', max_iter=10000)
    model_test.fit(X[train], y.values[train])
    y_hat = model_test.predict_proba(X[test])
    auc.append(roc_auc_score(y.values[test], y_hat[:,1]))
np.mean(auc)

kf = KFold(n_splits=10, shuffle=True)
auc = []
X = agg_data
y = cardio_risk
for train, test in kf.split(X):
    model_test = RandomForestClassifier(n_estimators=1000, min_samples_split=10)
    model_test.fit(X.values[train], y.values[train])
    y_hat = model_test.predict_proba(X.values[test])
    auc.append(roc_auc_score(y.values[test], y_hat[:,1]))
np.mean(auc)

kf = KFold(n_splits=10, shuffle=True)
auc = []
X = agg_data
y = cardio_risk
for train, test in kf.split(X):
    model_test = GradientBoostingClassifier(n_estimators=1000, max_depth=2)
    model_test.fit(X.values[train], y.values[train])
    y_hat = model_test.predict_proba(X.values[test])
    auc.append(roc_auc_score(y.values[test], y_hat[:,1]))
np.mean(auc)

kf = KFold(n_splits=10, shuffle=True)
auc = []
X = agg_data
y = cardio_risk
for train, test in kf.split(X):
    model_test = MLPClassifier(hidden_layer_sizes=(100, 2), activation='logistic',alpha=0.0001, max_iter=200)
    model_test.fit(X.values[train], y.values[train])
    y_hat = model_test.predict_proba(X.values[test])
    auc.append(roc_auc_score(y.values[test], y_hat[:,1]))
np.mean(auc)

#compare feature importance between logistic and RF
log_features = pd.DataFrame({'Fraction of Samples Affected' : np.absolute(model_log.coef_[0])}, 
                            index=agg_data.columns)
log_features = log_features.sort_values(by='Fraction of Samples Affected')

rf_features = pd.DataFrame({'Fraction of Samples Affected' : model_rf.feature_importances_},
                           index=agg_data.columns)
rf_features = rf_features.sort_values(by='Fraction of Samples Affected')

fig, ax = plt.subplots(1,2, figsize=(10,20))
ax[0].barh(log_features.index, log_features['Fraction of Samples Affected'].values, height=0.8)
ax[1].barh(rf_features.index, rf_features['Fraction of Samples Affected'].values, height=0.8)
ax[0].set_xlabel('Coefficient Magnitude')
ax[1].set_xlabel('Fraction of Samples Affected')
ax[0].set_title('Logistic Regression Feature Weights')
ax[1].set_title('Random Forest Feature Weights')
fig.tight_layout()
#plt.savefig('img/feature_importance.png')

#make ROC curves
fpr_log, tpr_log, thresh_log = roc_curve(y_test, y_hat_log[:,1])
fpr_rf, tpr_rf, thresh_rf = roc_curve(y_test, y_hat_rf[:,1])
fpr_gbc, tpr_gbc, thresh_gbc = roc_curve(y_test, y_hat_gbc[:,1])
fpr_neural, tpr_neural, thresh_neural = roc_curve(y_test, y_hat_neural[:,1])

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr_log, y=tpr_log,
                    mode='lines',
                    name='logistic, AUC=0.85',
                    line=dict(color='red', width=3)))
fig.add_trace(go.Scatter(x=fpr_neural, y=tpr_neural,
                    mode='lines',
                    name='neural network, AUC=0.84',
                    line=dict(color='orange', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf,
                    mode='lines',
                    name='random forest, AUC=0.84',
                    line=dict(color='green', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=fpr_gbc, y=tpr_gbc,
                    mode='lines',
                    name='gradient boosting, AUC=0.83',
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


def get_ypred(y_true, y_pred, threshold=0.5):
    '''
    Takes output from predict_proba and returns class predictions at given threshold.
    '''
    y_thresh = [1 if y_pred[i,1] > threshold else 0 for i in range(len(y_pred))]
    return np.array(y_thresh)

#create confusion matrix for logistic regression at threshold of 0.75
cm = confusion_matrix(y_test, get_ypred(y_test, y_hat_log, 0.75))
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cm, cmap='twilight')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1))
ax.yaxis.set(ticks=(0, 1))
ax.set_xticklabels(('Predicted 0', 'Predicted 1'), size='x-large')
ax.set_yticklabels(('Actual 0', 'Actual 1'), size='x-large')
ax.set_ylim(1.5, -0.5)
ax.set_title('Logistic, Threshold=0.75')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black', size='large')
plt.show()
#plt.savefig('img/cf_log.png')

#cost matrix at 0.75 threshold
cb = np.array([[0, -500],[-5000, 2000]]) * cm
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(cb, cmap='twilight')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1))
ax.yaxis.set(ticks=(0, 1))
ax.set_xticklabels(('Predicted 0', 'Predicted 1'), size='x-large')
ax.set_yticklabels(('Actual 0', 'Actual 1'), size='x-large')
ax.set_ylim(1.5, -0.5)
ax.set_title('Logistic Cost, Threshold=0.75')
for i in range(2):
    for j in range(2):
        ax.text(j, i, '${}'.format(cb[i, j]), ha='center', va='center', color='black', size='large')
plt.show()
#plt.savefig('img/cf_log.png')

# high-risk, correctly identified----------2,000
# high-risk, incorrectly indentified-------(-5,000)
# low-risk, correctly identified-----------0
# low-risk, incorrectly identified---------(-500)

profit = cb.sum()
