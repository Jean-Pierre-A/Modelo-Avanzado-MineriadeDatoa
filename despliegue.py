import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 

data = pd.read_csv("testDef.csv", low_memory=False)

import pickle
filename = 'modelo-clas.pkl'
model,labelencoder,variables,min_max_scaler = pickle.load(open(filename, 'rb'))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

data['Occupation']=data['Occupation'].astype('category')
data['Type_of_Loan']=data['Type_of_Loan'].astype('category')
data['Credit_Mix']=data['Credit_Mix'].astype('category')
data['Payment_of_Min_Amount']=data['Payment_of_Min_Amount'].astype('category')

data = pd.get_dummies(data, columns=['Occupation','Type_of_Loan','Credit_Mix','Payment_of_Min_Amount'], drop_first=False)

data=data.reindex(columns=variables,fill_value=0)

min_max_scaler.fit(data[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan','Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']]) #Ajuste de los parametros: max - min

#Se aplica la normalización a 70%  y 30%
data[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan','Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']]= min_max_scaler.transform(data[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan','Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']]) 
print(data.head())
print(data.info())

Y_fut = model.predict(data)
print(Y_fut)

print(labelencoder.inverse_transform(Y_fut))

