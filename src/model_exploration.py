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

# model = RandomForestClassifier(n_estimators=1000)
# for i in range(2,len(feat_scores)+1):
#     model.fit(agg_data[feat_scores[-i:].index], cardio_risk)
#     y_hat = model.predict_proba(agg_data[feat_scores[-i:].index])
#     plt.scatter(i, log_loss(cardio_risk, y_hat))

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

def plot_roc(ax, df):
    ax.plot([1]+list(df.fpr), [1]+list(df.tpr), label="ROC")
    ax.plot([0,1],[0,1], 'k', label="random")
    ax.set_xlabel('fpr')
    ax.set_ylabel('tpr')
    ax.set_title('ROC Curve')
    ax.legend()

fig, ax = plt.subplots()
df = calculate_threshold_values(y_prob[:,1], y_test)
plot_roc(ax, df)

roc_auc_score(y_test, y_prob[:,1])  #logistic regression
roc_auc_score(y_test, y_prob_scaled[:,1])  #logistic regression scaled
roc_auc_score(y_test, y_hat_rf[:,1]) #random forest
roc_auc_score(y_test, y_hat_gbc[:,1])    #gradient boosting classifier

# kf = KFold(n_splits=10, shuffle=True)
# fold_losses = []

# X = agg_data
# y = cardio_risk
# for train, test in kf.split(X):
#     model = LogisticRegression(C=1000)
#     model.fit(X.values[train], y.values[train])
#     y_hat = model.predict_proba(X.values[test])
#     fold_losses.append(log_loss(y.values[test], y_hat))
# fold_losses

