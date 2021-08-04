import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

df1 = pd.read_csv('healthcare-dataset-stroke-data.csv')

le = LabelEncoder()

df1['gender'] = le.fit_transform(df1.gender)
df1['ever_married'] = le.fit_transform(df1.ever_married)
df1['work_type'] = le.fit_transform(df1.work_type)
df1['Residence_type'] = le.fit_transform(df1.Residence_type)
df1['smoking_status'] = le.fit_transform(df1.smoking_status)

df1['bmi'] = df1.bmi.fillna(0)

x = df1.drop(df1[['id', 'stroke']], axis = 1)

y = df1[['stroke']]

classifier = RandomForestClassifier(n_estimators = 200, max_features = "log2", criterion= "gini", class_weight = "balanced")

classifier.fit(x,y.values.ravel())

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))