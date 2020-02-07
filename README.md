# Predicting Risk of Cardiovascular Disease

## Introduction

Cardiovascular disease continues to be the leading cause of death in the U.S.  Nearly half of all heart attacks have symptoms so mild that individuals don't know they are having a heart attack--a so-called silent heart attack.  Health professionals estimate that 8 to 11 million people suffer from silent strokes each year in which individuals are asymptomatic but would have evidence of a stroke on an MRI.

These risks, combined with the ever-increasing cost of healthcare in the U.S., indicate a need for increased diagnostic efficiency.  How can we identify the individuals who are most at risk?  What preventative measures could be implemented to decrease risk?

## Data

NHANES (National Health and Nutrition Examination Survey) is a naitonal survey conducted by the CDC every couple of years.  The survey contains over 1,000 variables, asking questions about lifestyle and medical history as well as conducting brief medical examinations and running blood tests.
The data used in building this model comes from the 2015-2016 survey and can be found at:  
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015.

The data has been limited to the adult population who participated in both the questionnaire and examination portions of the survey, resulting in 5,735 individuals.  Of the many variables available for analysis, the list was narrowed to 46 variables to use as features in this model.  The elimination process was based first on intuitive relevance to heart disease (e.g. excluding dental health) and second on quality of data (e.g. data with 5,000 missing values).

## Models

Individuals in the dataset have been labeled as high-risk for cardiovascular disease based on either:
1. A combination of answers to the Cardiovascular Health questionnaire which indicate history of angina, or
2. Answering the Medical History questionnaire in the affirmative for history of coronary heart disease, angina, heart attack or stroke 
Note: The Cardiovascular Health questionnaire was only administerd to adults age 40+.  The Medical History questionnaire was only administerd to adults age 20+.

While the survey includes data for people of all ages, only data for adults (18+) have been considered here.
Data has been excluded for individuals who did not complete the medical examinations (257 adults for 2016).
Some features have been omitted due to a very high percentage of missing values or a very high class imbalance.

Initial model: Logistic Regression (without train-test-split) based on age and gender; hard classification had no value (all individuals predicted as low-risk); soft classification resulted in log_loss=0.29

After all data cleaning and feature engineering was completed, possible models were applied to the entire featur matirix and were examined using train-test-split.  A Logistic Regression model was applied and the new log loss was 0.25 and ROC AUC of 0.85, with and without scaling.

A Random Forest Classifier model with n_estimators=1000 was explored next. The log loss was 0.25 and ROC AUC of 0.84. 

Generated feature importance plot from Random Forest.

Gradient Boosting with n_estimators=1000 had a log loss of 0.31 and ROC AUC of 0.82.

Adding diabetes indicator showed no difference.

![ROC curves for various models](img/roc_comparison.png)