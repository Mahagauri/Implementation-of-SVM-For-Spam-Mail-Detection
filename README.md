# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MAHAGAURI P
RegisterNumber:  212224040181
*/
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![SVM For Spam Mail Detection](sam.png)
## Encoding:
![image](https://github.com/user-attachments/assets/9b72e664-ce9c-4963-876e-96e6aaa461b4)


## Head():
![image](https://github.com/user-attachments/assets/93867079-59b8-4b72-b6b1-d7f2faaa5e42)



## Info():
![image](https://github.com/user-attachments/assets/4b7e0ab0-348e-4748-93b6-4564e898b757)



## isnull().sum():
![image](https://github.com/user-attachments/assets/b395f3fd-f262-4359-aaac-ffcb6acba107)



## Prediction of y:
![image](https://github.com/user-attachments/assets/d1bf70cd-cbce-4591-9a05-bb390897332b)



## Accuracy:
![image](https://github.com/user-attachments/assets/9c5c58de-ab35-4131-b9db-00b603c05157)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
