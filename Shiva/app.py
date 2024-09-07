import pandas as pd
data = pd.read_csv(r"C:\Users\5525\Downloads\Shiva\thermal_stress_testing_ic_dataset_with_accurate_suggestions(1).csv")


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Thermal Stability (Pass/Fail)'] = le.fit_transform(data['Thermal Stability (Pass/Fail)'])

from sklearn.model_selection import train_test_split
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.40, random_state=2000)

from sklearn.naive_bayes import GaussianNB
modelnb = GaussianNB()
modelnb.fit(xtrain, ytrain)
ypred = modelnb.predict(xtest)

from sklearn.metrics import accuracy_score
f1 = accuracy_score(ypred, ytest)
print(f1 * 100)
import joblib  
joblib.dump(modelnb, 'naive_bayes_model.pkl')
print("Model saved successfully!")
