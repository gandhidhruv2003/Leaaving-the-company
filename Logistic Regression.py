import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("HR_comma_sep.csv")

left = df[df.left==1]
retain = df[df.left==0]
new_df = df.groupby("left").mean()

pd.crosstab(df.salary, df.left).plot(kind='bar')
plt.show()

pd.crosstab(df.satisfaction_level, df.left).plot(kind="bar")
plt.show()

pd.crosstab(df.last_evaluation, df.left).plot(kind="bar")
plt.show()

pd.crosstab(df.number_project, df.left).plot(kind="bar")
plt.show()

pd.crosstab(df.average_montly_hours, df.left).plot(kind="bar")
plt.show()

pd.crosstab(df.time_spend_company, df.left).plot(kind="bar")
plt.show()

pd.crosstab(df.Work_accident, df.left).plot(kind="bar")
plt.show()

pd.crosstab(df.promotion_last_5years, df.left).plot(kind="bar")
plt.show()

pd.crosstab(df.Department, df.left).plot(kind="bar")
plt.show()

le = LabelEncoder()
df.salary = le.fit_transform(df.salary)

X = df[["satisfaction_level", "average_montly_hours", "promotion_last_5years", "salary"]]
Y = df.left

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

reg = LogisticRegression()
reg.fit(X_train, Y_train)
print(reg.predict(X_test))
print(Y_test)
print(reg.score(X_test, Y_test))