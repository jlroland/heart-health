from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

#Initial baseline model using age & gender
X = agg_data[['RIDAGEYR', 'RIAGENDR']]
y = np.array(cardio_risk)
model = LogisticRegression(C=1000)
model.fit(X,y)
y_prob = model.predict_proba(X)
log_loss(y, y_prob)

model = LogisticRegression(C=1000)
model.fit(agg_data, cardio_risk)
y_prob_all = model.predict_proba(agg_data)
log_loss(cardio_risk, y_prob_all)

model_rf = RandomForestClassifier(n_estimators=1000)
model_rf.fit(agg_data, cardio_risk)
y_hat = model_rf.predict_proba(agg_data)
log_loss(cardio_risk, y_hat)

feat_scores = pd.DataFrame({'Fraction of Samples Affected' : model_rf.feature_importances_},
                           index=agg_data.columns)
feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected')
feat_scores.plot(kind='barh', figsize=(10,20))

model_gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1000)
model_gbc.fit(agg_data, cardio_risk)
y_gbc = model_gbc.predict_proba(agg_data)
log_loss(cardio_risk, y_gbc)

# model = RandomForestClassifier(n_estimators=1000)
# for i in range(2,len(feat_scores)+1):
#     model.fit(agg_data[feat_scores[-i:].index], cardio_risk)
#     y_hat = model.predict_proba(agg_data[feat_scores[-i:].index])
#     plt.scatter(i, log_loss(cardio_risk, y_hat))

roc_auc_score(cardio_risk, y_prob[:,1])  #logistic regression
roc_auc_score(cardio_risk, y_hat_rf[:,1]) #random forest
roc_auc_score(cardio_risk, y_gbc[:,1])    #gradient boosting classifier
