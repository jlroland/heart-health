import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss, roc_curve, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

age_gender = agg_data[['age', 'gender']]
colors = np.where(cardio_risk==1,'r','g')
fig,ax = plt.subplots(figsize=(10,8))
#age_gender.plot.scatter(x='age',y='gender', alpha=0.01, c=colors, title='Risk Distribution for Age & Gender',
#                       figsize=(10,8))
#plt.show()
ax.scatter(x=agg_data['age'][cardio_risk==0],y=agg_data['gender'][cardio_risk==0], alpha=0.2, c='green', 
           label='low-risk')
ax.scatter(x=agg_data['age'][cardio_risk==1],y=agg_data['gender'][cardio_risk==1], alpha=0.2, c='red', 
           label='high-risk')
ax.set_xlabel('Age (years)')
ax.set_yticks([0,1])
ax.set_yticklabels(['female', 'male'])
ax.set_ylabel('Gender')
ax.set_title('Risk Distribution by Age & Gender')
ax.legend()
plt.savefig('img/initial_model_dist.png')

#Initial baseline model using age & gender
X_train_init, X_test_init, y_train_init, y_test_init = train_test_split(agg_data[['age', 'gender']],cardio_risk)
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_init,y_train_init)
y_prob_init = model.predict_proba(X_test_init)
#log_loss(y_test_init, y_prob_init)
roc_auc_score(y_test_init, y_prob_init[:,1])

X_train, X_test, y_train, y_test = train_test_split(agg_data,cardio_risk)
scaler_train = MinMaxScaler()
scaler_test = MinMaxScaler()
X_train_scaled, X_test_scaled = scaler_train.fit_transform(X_train), scaler_test.fit_transform(X_test)

#feature scaling and regularization
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

#find best estimate of ROC AUC
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

fig.tight_layout()
#plt.savefig('img/rf_feature_importance.png')

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


def make_cf(y_true, y_pred, threshold=0.5):
    y_thresh = [1 if y_pred[i,1] > threshold else 0 for i in range(len(y_pred))]
    confusion = pd.DataFrame(confusion_matrix(y_test, y_thresh), index=['actual_0', 'actual_1'], 
                             columns=['predicted_0', 'predicted_1'])
    return confusion

# high-risk, correctly identified----------2,000
# high-risk, incorrectly indentified-------(-10,000)
# low-risk, correctly identified-----------0
# low-risk, incorrectly identified---------(-500)
cb_log = np.array([[0, -500],[-10000, 2000]]) * make_cf(y_test, y_hat_log, 0.5)
profit_log = cb_log.values.sum()

cb_neural = np.array([[0, -500],[-10000, 2000]]) * make_cf(y_test, y_hat_neural, 0.5)
profit_neural = cb_neuaral.values.sum()
