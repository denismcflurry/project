import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# imports and data

df = pd.read_csv("data/heart.csv")

print("columns:")
print(df.columns)

print("\ntargets:")
print(df["target"].value_counts())


# x and y

X = df.drop("target", axis=1)
y = df["target"]

cols = X.columns


# split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67
)


# logistic regression

log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)
log_error = 1 - log_acc


# random forest

forest = RandomForestClassifier(random_state=67)
forest.fit(X_train, y_train)

forest_pred = forest.predict(X_test)
forest_acc = accuracy_score(y_test, forest_pred)
forest_error = 1 - forest_acc


# model comparison

print("\nLogistic Regression Vs Random Forest")

print("\nLogistic Regression accuracy:", round(log_acc * 100, 2), "%")
print("Logistic Regression error:", round(log_error * 100, 2), "%")

print("\nRandom Forest accuracy:", round(forest_acc * 100, 2), "%")
print("Random Forest error:", round(forest_error * 100, 2), "%")

print("\nAccuracy difference:", round(abs(log_acc - forest_acc) * 100, 2), "%")
print("Error difference:", round(abs(log_error - forest_error) * 100, 2), "%")

if log_acc > forest_acc:
    print("Logistic Regression was better here")
elif forest_acc > log_acc:
    print("Random Forest was better here")
else:
    print("Both models had same accuracy")


# some extra checking

print("\nConfusion matrix for Random Forest:")
print(confusion_matrix(y_test, forest_pred))

print("\nClassification report for Random Forest:")
print(classification_report(y_test, forest_pred))


# prediction text

def result_text(x):
    if x == 1:
        return "Higher risk / heart disease signs"
    return "Lower risk / no heart disease signs"


def prob(model, pacient):
    index = list(model.classes_).index(1)
    return model.predict_proba(pacient)[:, index]


# manual patients

low_patient = pd.DataFrame([{
    "age": 35,
    "sex": 0,
    "cp": 0,
    "trestbps": 115,
    "chol": 180,
    "fbs": 0,
    "restecg": 1,
    "thalach": 170,
    "exang": 0,
    "oldpeak": 0.2,
    "slope": 2,
    "ca": 0,
    "thal": 2
}])

high_patient = pd.DataFrame([{
    "age": 62,
    "sex": 1,
    "cp": 3,
    "trestbps": 150,
    "chol": 280,
    "fbs": 1,
    "restecg": 0,
    "thalach": 110,
    "exang": 1,
    "oldpeak": 2.8,
    "slope": 0,
    "ca": 2,
    "thal": 3
}])

low_patient = low_patient[cols]
high_patient = high_patient[cols]


# manual test

print("\nManual fictional patients")

patients = [
    ("low risk patient", low_patient),
    ("high risk patient", high_patient)
]

for name, p in patients:
    print("\nPatient:", name)

    a = log_model.predict(p)[0]
    b = forest.predict(p)[0]

    a_prob = prob(log_model, p)[0]
    b_prob = prob(forest, p)[0]

    print("Logistic Regression:", a, "-", result_text(a), "-", round(a_prob * 100, 2), "%")
    print("Random Forest:", b, "-", result_text(b), "-", round(b_prob * 100, 2), "%")

    if a == b:
        print("models agree")
    else:
        print("models disagree")


# random patients generator

def make_random_patients(n):
    list_of_people = []

    for i in range(n):
        person = {
            "age": np.random.randint(25, 80),
            "sex": np.random.randint(0, 2),
            "cp": np.random.randint(0, 4),
            "trestbps": np.random.randint(90, 180),
            "chol": np.random.randint(120, 350),
            "fbs": np.random.randint(0, 2),
            "restecg": np.random.randint(0, 3),
            "thalach": np.random.randint(80, 200),
            "exang": np.random.randint(0, 2),
            "oldpeak": round(np.random.uniform(0, 5), 1),
            "slope": np.random.randint(0, 3),
            "ca": np.random.randint(0, 4),
            "thal": np.random.randint(0, 4)
        }

        list_of_people.append(person)

    data = pd.DataFrame(list_of_people)
    data = data[cols]

    return data


# random patients test

print("\nRandom patients generator")

try:
    n = input("How many random patients? 10 / 20 / 30: ")

    if n == "":
        n = 10
    else:
        n = int(n)

except:
    print("wrong input, using 10")
    n = 10


random_people = make_random_patients(n)

log_random = log_model.predict(random_people)
forest_random = forest.predict(random_people)

log_random_prob = prob(log_model, random_people)
forest_random_prob = prob(forest, random_people)

random_people["Logistic Regression"] = log_random
random_people["Logistic Risk %"] = [round(x * 100, 2) for x in log_random_prob]

random_people["Random Forest"] = forest_random
random_people["Forest Risk %"] = [round(x * 100, 2) for x in forest_random_prob]

agree = []

for i in range(n):
    if log_random[i] == forest_random[i]:
        agree.append("Agree")
    else:
        agree.append("Disagree")

random_people["Logistic Regression Vs Random Forest"] = agree

print("\nRandom fictional patients result:")
print(random_people)

print("\nAgreement count:")
print(random_people["Logistic Regression Vs Random Forest"].value_counts())

print("\nThis is only a student AI project, not a real medical diagnosis.")