## Final Modelo Avanzado Realizado por Jean Pierre Agudelo y Juan nicolas ruiz 
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
#print(data.describe())



print(data.describe())

data['Age'].value_counts().plot(kind='bar')
data['Num_Bank_Accounts'].value_counts().plot(kind='bar')


data.Age[data["Age"]<=0 ] = None
data.Age[data["Age"]>=100 ] = None
data.Num_Bank_Accounts[data["Num_Bank_Accounts"]<=0 ] = None
data.Num_Bank_Accounts[data["Num_Bank_Accounts"]>=10 ] = None
data.Num_Credit_Card[data["Num_Credit_Card"]<=0 ] = None
data.Num_Credit_Card[data["Num_Credit_Card"]>=10 ] = None
data.Num_of_Loan[data["Num_of_Loan"]<=0 ] = None
data.Num_of_Loan[data["Num_of_Loan"] >= 5 ] = None
data['Num_Bank_Accounts'].value_counts().plot(kind='bar')

#data.Age[data["Num_Bank_Accounts"]<=0 ] = None

# Mostrar el gráfico
#plt.show()
ImpNumeros = SimpleImputer(missing_values=np.nan, strategy='mean')
 
data[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan', 'Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']] = ImpNumeros.fit_transform(data[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan','Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']])

ImpCategoricas = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
 
data[['Occupation','Type_of_Loan','Credit_Mix','Payment_of_Min_Amount','Credit_Score']] = ImpCategoricas.fit_transform(data[['Occupation','Type_of_Loan','Credit_Mix','Payment_of_Min_Amount','Credit_Score']])

data = pd.get_dummies(data, columns=['Occupation','Type_of_Loan','Credit_Mix','Payment_of_Min_Amount'], drop_first=False)
labelencoder = LabelEncoder()

data["Credit_Score"]=labelencoder.fit_transform(data["Credit_Score"])

data.head()
print(data.describe())
print(data.info())

#División 70-30
X = data.drop("Credit_Score", axis = 1) # Variables predictoras
Y = data['Credit_Score'] #Variable objetivo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
Y_train.value_counts().plot(kind='bar')# Objetivo del 70%
plt.show()