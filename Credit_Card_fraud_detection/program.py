import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('creditcard.csv')

X=df.drop('Class',axis=1)
y=df['Class']

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X['sc_time']=sc.fit_transform(X['Time'].values.reshape(-1,1))
X['sc_amt']=sc.fit_transform(X['Amount'].values.reshape(-1,1))

X=X.drop(['Time','Amount'],axis=1)

#now splitting the data into training & testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#stratify=y preserves fraud distribution properly

# !pip install imbalanced-learn

from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=42)
X_train_resamp, y_train_resamp = sm.fit_resample(X_train,y_train)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(X_train_resamp,y_train_resamp)

y_pred=model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:,1]

from sklearn.metrics import classification_report
print("Classification Report of Logistic Regression: ")
print(classification_report(y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_resamp, y_train_resamp)

y_pred_rf=rf_model.predict(X_test)
y_pred_prob_rf=rf_model.predict_proba(X_test)[:,1]

print("Classification Report of Random Forest Classifier: ")
print(classification_report(y_test, y_pred_rf))

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_resamp, y_train_resamp)

y_pred_dt = dt_model.predict(X_test)
y_pred_prob_dt = dt_model.predict_proba(X_test)[:,1]

print("Classification Report of Decision Tree: ")
print(classification_report(y_test, y_pred_dt))

