# heart-health

Individuals in the dataset have been labeled as high-risk for cardiovascular disease based on either:
1. A combination of answers to the Cardiovascular Health questionnaire which indicate history of angina, or
2. Answering the Medical History questionnaire in the affirmative for history of coronary heart disease, angina, heart attack or stroke 
Note: The Cardiovascular Health questionnaire was only administerd to adults age 40+.  The Medical History questionnaire was only administerd to adults age 20+.

While the survey includes data for people of all ages, only data for adults (18+) have been considered here.
Data has been excluded for individuals who did not complete the medical examinations (257 adults for 2016).
Some features have been omitted due to a very high percentage of missing values or a very high class imbalance.

Initial model: Logistic Regression based on age and gender; hard classification had no value (all individuals predicted as low-risk); soft classification resulted in log_loss=0.29

After all data cleaning and feature engineering was completed, another Logistic Regression model was applied to the entire feature matrix. The new log loss was 0.25.

A Random Forest Classifier model with n_estimators=1000 was explored next. The entire feature matrix was used and the log loss was 0.07. 

Generated feature importance plot from Random Forest.

Gradient Boosting with n_estimators=1000 performed similarly to Random Forest.
