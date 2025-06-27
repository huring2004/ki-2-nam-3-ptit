
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('diabetes.csv')
# df.info()



# print("Nulls")
# print("=====")
# print(df.isnull().sum())

# check 0s
# print("0s")
# print("+++")
# print(df.eq(0).sum())

df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
df.fillna(df.mean(),inplace=True)
# print(df.eq(0).sum())




corr = df.corr()
# fig, ax = plt.subplots(figsize=(10, 10))
# cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,len(df.columns),1)
# ax.set_xticks(ticks)
# ax.set_xticklabels(df.columns)
# plt.xticks(rotation = 90)
# ax.set_yticklabels(df.columns)
# ax.set_yticks(ticks)
# #---print the correlation factor---
# for i in range(df.shape[1]):
#  for j in range(9):
#   text = ax.text(j, i, round(corr.iloc[i][j],2), ha="center", va="center", color="w")
# plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 8))  # Thiết lập kích thước mới
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
#
# plt.show()  # Hiển thị heatmap mới

# print(df.corr().nlargest(4,'Outcome').values[:,8])


from sklearn import linear_model
from sklearn.model_selection import cross_val_score


#---features---
X = df[['Glucose','BMI','Age']]
#---label---
y = df.iloc[:,8]

# print(y)
# log_regress = linear_model.LogisticRegression()
# log_regress_score = cross_val_score(log_regress, X, y, cv=10,scoring='accuracy').mean()


# result = []
# result.append(log_regress_score)
# print(log_regress_score)
#
#
from sklearn.neighbors import KNeighborsClassifier
# #---empty list that will hold cv (cross-validates) scores---




# cv_scores = []
# folds = 10
# ks = list(range(1,int(len(X) * ((folds - 1)/folds)), 2))
# for k in ks:
#  knn = KNeighborsClassifier(n_neighbors=k)
#  score = cross_val_score(knn, X, y, cv=folds, scoring='accuracy').mean()
#  cv_scores.append(score)
# knn_score = max(cv_scores)
# optimal_k = ks[cv_scores.index(knn_score)]
# print(f"The optimal number of neighbors is {optimal_k}")
# print(knn_score)



# result.append(knn_score)
#
#
#
from sklearn import svm
# linear_svm = svm.SVC(kernel='linear')
# linear_svm_score = cross_val_score(linear_svm, X, y,
#  cv=10, scoring='accuracy').mean()
# print(linear_svm_score)


# result.append(linear_svm_score)
#


# rbf = svm.SVC(kernel='rbf')
# rbf_score = cross_val_score(rbf, X, y, cv=10, scoring='accuracy').mean()
# print(rbf_score)

#
#

#
#
import pickle

knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X, y)
filename = 'diabetes.sav'
pickle.dump(knn, open(filename, 'wb'))

#---load the model from disk---
loaded_model = pickle.load(open(filename, 'rb'))

Glucose = 40
BMI = 43.1
Age = 53
prediction = loaded_model.predict(pd.DataFrame([[Glucose, BMI, Age]]))

if prediction[0]==0:
 print("Non-diabetic")
else:
 print("Diabetic")

proba = loaded_model.predict_proba(pd.DataFrame([[Glucose, BMI, Age]], columns=['Glucose', 'BMI', 'Age']))
print(proba)
print("Confidence: " + str(round(np.amax(proba[0]) * 100, 2)) + "%")


