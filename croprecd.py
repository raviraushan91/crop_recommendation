import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

crop=pd.read_csv("Crop_recommendation.csv")
# print(crop.head())
# print(crop.info)
# print(dft.shape)
# print(dft.isnull().sum())
# print(dft.duplicated().sum())
# print(dft.describe())
# print(crop.corr())
# print(sns.heatmap(crop.corr(),annot=True,cbar=True))
# print(crop.label.value_counts())
# print(crop['label'].unique.size)
# sns.distplot(crop['P'])
# print(plt.show())
# sns.distplot(crop['N'])
# print(plt.show())
crop_dict={
    
'maize      ':1,
'chickpea   ':2,
'kidneybeans':3,
'pigeonpeas ':4,
'mothbeans  ':5,
'mungbean   ':6,
'blackgram  ':7,
'lentil     ':8,
'pomegranate':9,
'banana     ':10,
'mango      ':11,
'grapes     ':12,
'watermelon ':13,
'muskmelon  ':14,
'apple      ':15,
'orange     ':16,
'papaya     ':17,
'coconut    ':18,
'cotton     ':19,
'jute       ':20,
'rice':21
    
}
crop['label']=crop['label'].map(crop_dict)
# print(crop.head())
# print(crop.label.unique())
# print(crop.value_counts)
X=crop.drop('label',axis=1)
Y=crop['label']
# print(X.head())
# print(Y.head())

mask=~np.isnan(Y)
X=X[mask]
Y=Y[mask]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train.shape)

mx=MinMaxScaler()
X_train=mx.fit_transform(X_train)
X_test=mx.transform(X_test)
# print(X_train)

sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)
# print(X_test)
# import all models frm scirpt-learn

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score

# create  a dictionay of all models

models={
    'LogisticRegression':LogisticRegression(),
    'GaussianNB':GaussianNB(),
    'SVC':SVC(),
    'KNeighborsClassifier':KNeighborsClassifier(),
    'DecisionTreeClassifier':DecisionTreeClassifier(),
    'ExtraTreeClassifier':ExtraTreeClassifier(),
    'RandomForestClassifier':RandomForestClassifier(),
    'BaggingClassifier':BaggingClassifier(),
    'GradientBoostingClassifier':GradientBoostingClassifier(),
    'AdaBoostClassifier':AdaBoostClassifier()
    }

# for name ,xyz in models.items():
#       #trained the model 
#     xyz.fit(X_train,Y_train)
#      #target the model 
    
#     Y_pred=xyz.predict(X_test)
#     # calculate the accuracy
#     score=accuracy_score(Y_test,Y_pred)
#     # print the model name and its accuracy
#     print(f"{name} model with accuracy : {score:.2f}")
    
# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, Y_train)
    
    # Predict using the test set
    Y_pred = model.predict(X_test)
    
    # Calculate accuracy
    score = accuracy_score(Y_test, Y_pred)
    
    # Print model name and accuracy
    print(f"{name} model with accuracy: {score}")
    
xyzclf=DecisionTreeClassifier()
xyzclf.fit(X_train,Y_train)
Y_pred=xyzclf.predict(X_test)
accuracy_score(Y_test,Y_pred)
print(crop.columns)
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    mx_features=mx.fit_transform(features)
    sc_mx_features = sc.fit_transform(mx_features)
    prediction = xyzclf.predict(sc_mx_features).reshape(1,-1)
    return prediction[0]

# print(crop.head())
N=74
P=35
K=40
temperature=26.49
humidity=81.15
ph=6.9
rainfall=242.71
predict = recommendation(N,P,K,temperature,humidity,ph,rainfall)
print(predict) 
# import pickle
# pickle.dump(xyzclf,open('model.pkl','wb'))
# pickle.dump(mx,open('minmaxscaler.pkl','wb'))
# pickle.dump(sc,open('standscaler.pkl','wb'))

