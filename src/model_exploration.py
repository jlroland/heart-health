from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

#Initial baseline model using age & gender
X = agg_data[['RIDAGEYR', 'RIAGENDR']]
y = np.array(cardio_risk)
model = LogisticRegression(C=1000)
model.fit(X,y)
y_prob = model.predict_proba(X)
log_loss(y, y_prob)