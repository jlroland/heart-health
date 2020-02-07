from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Initial baseline model using age & gender
X_init = agg_data[['RIDAGEYR', 'RIAGENDR']]
y_init = np.array(cardio_risk)
model_init = LogisticRegression(C=1000)
model_init.fit(X_init,y_init)
y_prob_init = model.predict_proba(X_init)
log_loss(y_init, y_prob_init)

X_train, X_test, y_train, y_test = train_test_split(agg_data,cardio_risk)
scaler_train = MinMaxScaler()
scaler_test = MinMaxScaler()
X_train_scaled, X_test_scaled = scaler_train.fit_transform(X_train), scaler_test.fit_transform(X_test)

model = LogisticRegression(C=1000)
model.fit(X_train,y_train)
y_prob = model.predict_proba(X_test)
log_loss(y_test, y_prob)

#try using only top 10 features by importance (based on coefficients)
ind = list(np.absolute(model.coef_).argsort()[0])
top10_log = agg_data.iloc[:,ind[-10:]]
X_train10_log, X_test10_log, y_train10_log, y_test10_log = train_test_split(top10_log, cardio_risk)
model10_log = LogisticRegression(C=1000, solver='lbfgs', max_iter=10000)
model10_log.fit(X_train10_log, y_train10_log)
y_hat10_log = model10_log.predict_proba(X_test10_log)
log_loss(y_test10_log, y_hat10_log)

model_scaled = LogisticRegression(C=1000)
model_scaled.fit(X_train_scaled,y_train)
y_prob_scaled = model_scaled.predict_proba(X_test_scaled)
log_loss(y_test, y_prob_scaled)

model_rf = RandomForestClassifier(n_estimators=1000)
model_rf.fit(X_train, y_train)
y_hat_rf = model_rf.predict_proba(X_test)
log_loss(y_test, y_hat_rf)

feat_scores = pd.DataFrame({'Fraction of Samples Affected' : model_rf.feature_importances_},
                           index=agg_data.columns)
feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected')
feat_scores.plot(kind='barh', figsize=(10,20))

model_gbc = GradientBoostingClassifier(n_estimators=1000)
model_gbc.fit(X_train, y_train)
y_hat_gbc = model_gbc.predict_proba(X_test)
log_loss(y_test, y_hat_gbc)

neural = MLPClassifier(hidden_layer_sizes=(100, 5), activation='logistic',alpha=0.0001, max_iter=200)
neural.fit(X_train, y_train)
y_hat_neural = neural.predict_proba(X_test)
log_loss(y_test, y_hat_neural)

roc_auc_score(y_test, y_prob[:,1])  #logistic regression
roc_auc_score(y_test, y_prob_scaled[:,1])  #logistic regression scaled
roc_auc_score(y_test, y_hat_rf[:,1]) #random forest
roc_auc_score(y_test, y_hat_gbc[:,1])    #gradient boosting classifier
roc_auc_score(y_test, y_hat_neural[:,1])  #neural network

def calculate_threshold_values(prob, y):
    '''
    Build dataframe of the various confusion-matrix ratios by threshold
    from a list of predicted probabilities and actual y values
    '''
    df = pd.DataFrame({'prob': prob, 'y': y})
    df.sort_values('prob', inplace=True)
    
    actual_p = df.y.sum()
    actual_n = df.shape[0] - df.y.sum()

    df['tn'] = (df.y == 0).cumsum()
    df['fn'] = df.y.cumsum()
    df['fp'] = actual_n - df.tn
    df['tp'] = actual_p - df.fn

    df['fpr'] = df.fp/(df.fp + df.tn)
    df['tpr'] = df.tp/(df.tp + df.fn)
    df['precision'] = df.tp/(df.tp + df.fp)
    df = df.reset_index(drop=True)
    return df

df1 = calculate_threshold_values(y_prob[:,1], y_test)
df2 = calculate_threshold_values(y_hat_rf[:,1], y_test)
df3 = calculate_threshold_values(y_hat_gbc[:,1], y_test)
df4 = calculate_threshold_values(y_hat_neural[:,1], y_test)

fig = go.Figure()
fig.add_trace(go.Scatter(x=[1]+list(df1.fpr), y=[1]+list(df1.tpr),
                    mode='lines',
                    name='logistic, AUC=0.86',
                    line=dict(color='red', width=4)))
fig.add_trace(go.Scatter(x=[1]+list(df4.fpr), y=[1]+list(df4.tpr),
                    mode='lines',
                    name='neural network, AUC=0.85',
                    line=dict(color='orange', width=4, dash='dash')))
fig.add_trace(go.Scatter(x=[1]+list(df2.fpr), y=[1]+list(df2.tpr),
                    mode='lines',
                    name='random forest, AUC=0.84',
                    line=dict(color='green', width=4, dash='dash')))
fig.add_trace(go.Scatter(x=[1]+list(df3.fpr), y=[1]+list(df3.tpr),
                    mode='lines',
                    name='gradient boosting, AUC=0.83',
                    line=dict(color='blue', width=4, dash='dash')))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                    mode='lines',
                    name='random',
                    line=dict(color='black', width=3, dash='dash')))

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
cb = np.array([[0, -500],[-10000, 2000]]) * make_cf(y_test, y_prob, 0.5)
profit = cb.values.sum()

# kf = KFold(n_splits=5, shuffle=True)
# auc = []

# X = agg_data
# y = cardio_risk
# for train, test in kf.split(X):
#     model = LogisticRegression(C=1000, solver='lbfgs', max_iter=10000)
#     model.fit(X.values[train], y.values[train])
#     y_hat = model.predict_proba(X.values[test])
#     auc.append(roc_auc_score(y.values[test], y_hat[:,1]))
# np.mean(auc)

model = MLPClassifier()
parameters = {'hidden_layer_sizes': (range(1, 101), range(1, 101)), 'activation': ['logistic', 'tanh', 'relu'], 'alpha': [0.001, 0.005, 0.01], 'max_iter': [100, 200, 500]}
grid = GridSearchCV(model, parameters, scoring='roc_auc')
grid.fit(X_train, y_train)
y_hat_grid = grid.predict_proba(X_test)
log_loss(y_test, y_hat_grid), roc_auc_score(y_test, y_hat_grid[:,1])
