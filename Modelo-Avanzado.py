## Final Modelo Avanzado Realizado por Jean Pierre Agudelo y Juan nicolas ruiz 
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 

#Integración de los datos
data = pd.read_csv("Credit Score.csv", low_memory=False)
#data = pd.read_excel("Credit score classification.xlsx",sheet_name=0)
#Configuramos los parámetros para que imprima de forma completa
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


data = data.drop(['ID','Customer_ID','Month','Name','SSN','Monthly_Inhand_Salary','Delay_from_due_date','Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries','Total_EMI_per_month','Amount_invested_monthly','Interest_Rate','Credit_History_Age','Payment_Behaviour'],axis=1)

data['Occupation']=data['Occupation'].astype('category')
data['Type_of_Loan']=data['Type_of_Loan'].astype('category')
data['Credit_Mix']=data['Credit_Mix'].astype('category')
data['Payment_of_Min_Amount']=data['Payment_of_Min_Amount'].astype('category')
data['Credit_Score']=data['Credit_Score'].astype('category')


print(data.info())
