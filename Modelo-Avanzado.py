## Final Modelo Avanzado Realizado por Jean Pierre Agudelo y Juan nicolas ruiz 
import pandas as pd 
import numpy as np  
import pickle
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors  import KNeighborsClassifier 

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

print("-------- Cambio de tipo de variable a categórica --------")
print(data.info())

#Estadística descriptiva
print("-------- Descripción de variables numéricas --------")
print(data.describe())

print("-------- Gráfica de variable edad --------")
#data['Age'].value_counts().plot(kind='bar')
print("-------- Gráfica de numero de cuentas de banco --------")
#data['Num_Bank_Accounts'].value_counts().plot(kind='bar')

#Limpieza de Atípicos
print("-------- Limpieza de atípicos --------")
data.Age[data["Age"]<=0 ] = None
data.Age[data["Age"]>=100 ] = None
data.Num_Bank_Accounts[data["Num_Bank_Accounts"]<=0 ] = None
data.Num_Bank_Accounts[data["Num_Bank_Accounts"]>=10 ] = None
data.Num_Credit_Card[data["Num_Credit_Card"]<=0 ] = None
data.Num_Credit_Card[data["Num_Credit_Card"]>=10 ] = None
data.Num_of_Loan[data["Num_of_Loan"]<=0 ] = None
data.Num_of_Loan[data["Num_of_Loan"] >= 5 ] = None

data['Occupation'].value_counts().plot(kind='bar', title="Occupation")
plt.show()

data['Type_of_Loan'].value_counts().plot(kind='bar', title="Tipos Type_of_Loan préstamos")
plt.show()

data['Credit_Mix'].value_counts().plot(kind='bar', title="Credit Mix")
plt.show()

data['Payment_of_Min_Amount'].value_counts().plot(kind='bar', title="Payment_of_Min_Amount")
plt.show()

data['Age'].plot(kind='box')
plt.show()

data['Num_Bank_Accounts'].plot(kind='box')
plt.show()

data['Num_Credit_Card'].plot(kind='box')
plt.show()

data['Num_of_Loan'].plot(kind='box')
plt.show()
#Imputación de nulos

print("-------- Imputación de nulos --------")
ImpNumeros = SimpleImputer(missing_values=np.nan, strategy='mean')
 
data[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan', 'Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']] = ImpNumeros.fit_transform(data[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan','Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']])

ImpCategoricas = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
 
data[['Occupation','Type_of_Loan','Credit_Mix','Payment_of_Min_Amount','Credit_Score']] = ImpCategoricas.fit_transform(data[['Occupation','Type_of_Loan','Credit_Mix','Payment_of_Min_Amount','Credit_Score']])


#Creación de dummies
print("-------- Creación de dummies --------")
data = pd.get_dummies(data, columns=['Occupation','Type_of_Loan','Credit_Mix','Payment_of_Min_Amount'], drop_first=False)
#Encoder de variable ojbetivo
print("-------- Encoder de variable objetivo --------")
labelencoder = LabelEncoder()
data["Credit_Score"]=labelencoder.fit_transform(data["Credit_Score"])

print("-------- Estado de los datos después de la preparación --------")
#print(data.describe())
print(data.info())

#División 70-30
print("-------- División 70-30 --------")
X = data.drop("Credit_Score", axis = 1) # Variables predictoras
Y = data['Credit_Score'] #Variable objetivo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
Y_test.value_counts().plot(kind='bar',title="Y test")# Objetivo del 70%
#plt.show()

print("-------- Balanceo de los datos --------")
from imblearn.over_sampling import RandomOverSampler
#oversample = RandomOverSampler(sampling_strategy='minority')
#Balanceo de datos
from imblearn.over_sampling import SMOTENC, SMOTE

#Balanceo para variables predictoras con al menos una categoría
sm = SMOTENC(categorical_features=[1,3]) #se indican las variables categoricas

#Balanceo para variables predictoras numéricas
#sm = SMOTE() 
X_train, Y_train = sm.fit_resample(X_train,Y_train) #Se almacenan el resultado en las mismas variables

#X_train,Y_train = oversample.fit_resample(X_train, Y_train)
#X_train,Y_train = oversample.fit_resample(X_train, Y_train)

Y_train.value_counts().plot(kind='bar', title="SMOTE")# Objetivo del 70%
#plt.show()

print(data.head())
print("-------- Normalización --------")
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(data[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan','Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']]) #Ajuste de los parametros: max - min

#Se aplica la normalización a 70%  y 30%
X_train[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan','Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']]= min_max_scaler.transform(X_train[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan','Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']]) #70%
X_test[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan','Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']]= min_max_scaler.transform(X_test[['Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card','Num_of_Loan','Outstanding_Debt','Credit_Utilization_Ratio','Monthly_Balance']])  #30%
print(data.head())
print("-------- Validación cruzada --------")
print("-------- Tree --------")
#Método de ML a usar en la validación cruzada
modelTree = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=2, max_depth=10)
resultadosTree = pd.DataFrame()
scores = cross_validate(modelTree, X_train, Y_train, cv=10, scoring=('accuracy','precision_macro','recall_macro'), return_train_score=False, return_estimator=False)
resultadosTree=pd.DataFrame(scores)
#resultadosTree=resultadosTree.rename(columns={'test_accuracy':'Tree_accuracy','test_precision_macro':'Tree_precision','test_recall_macro':'Tree_recall'})
#print(resultadosTree)

print("--------Media-------")
#print(resultadosTree.mean())

medidas = pd.DataFrame()
medidas['Tree', 'Tree2'] = resultadosTree.mean()

resultadosTree.describe()

print("-------- Neural Network --------")
from sklearn.neural_network import MLPClassifier
resultadosNN = pd.DataFrame()
modelNN = MLPClassifier(activation="logistic",hidden_layer_sizes=(5), learning_rate='constant',
                     learning_rate_init=0.2, momentum= 0.3, max_iter=5000, random_state=3)
scores = cross_validate(modelNN, X_train, Y_train, cv=10, scoring=('accuracy','precision_macro','recall_macro'), return_train_score=False, return_estimator=False)
resultadosNN = pd.DataFrame(scores)
#resultadosNN=resultadosNN.rename(columns={'test_accuracy':'NN_accuracy','test_precision_macro':'NN_precision','test_recall_macro':'NN_recall'})
#print(resultadosNN)
medidas['NN1', 'NN22'] = resultadosNN.mean()
         
print("-------- KNN--------")

modelKnn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
scores = cross_validate(modelKnn, X_train, Y_train, cv=10, scoring=('accuracy','precision_macro','recall_macro'), return_train_score=False, return_estimator=False)
resultadosKNN = pd.DataFrame(scores)
#resultadosKNN=resultadosKNN.rename(columns={'test_accuracy':'KNN_accuracy','test_precision_macro':'KNN_precision','test_recall_macro':'KNN_recall'})
#print(resultadosKNN)
medidas['KNN', 'KNN2'] = resultadosKNN.mean()

from  sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
scores = cross_validate(LogReg, X_train, Y_train, cv=10, scoring=('accuracy','precision_macro','recall_macro'), return_train_score=False, return_estimator=False)
resultadosLogReg = pd.DataFrame(scores)
#resultadosLogReg=resultadosLogReg.rename(columns={'test_accuracy':'LG_accuracy','test_precision_macro':'LG_precision','test_recall_macro':'LG_recall'})
#print(resultadosLogReg)
medidas['LogReg', 'LogReg2'] = resultadosLogReg.mean()

print(medidas)

print("----  Mejores 2 modelos Arboles y Knn ------ ")
print("Hiperparametrización - Arboles")
#  Arbol

modelTree = DecisionTreeClassifier()

# Definir los hiperparametros
criterion=['entropy','gini'] #Indice de información
min_samples_leaf=[2,10,50,100] # Cantidad de registros por hoja
max_depth=[None, 10,20, 50] #Niveles de profundidad

#Grid
param_grid = dict(criterion=criterion, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
grid = GridSearchCV(estimator=modelTree, param_grid=param_grid, scoring='f1_macro', n_jobs=-1, cv=10)
grid.fit(X_train, Y_train)

#Mejor modelo
modelTree= grid.best_estimator_

#Medida de evaluación del mejor modelo
medidasH= pd.DataFrame(index=['f1 de la CV'])
medidasH['Tree']=grid.best_score_

# Mejores párametros
print( grid.best_params_)
print(medidasH)

#KNN
print("Hiperparametrización - KNN")
modelKnn = KNeighborsClassifier()

# Definir los hiperparametros
n_neighbors=[1,3,5,7,8]
metric=['euclidean','minkowski']

#Grid
param_grid = dict(n_neighbors=n_neighbors, metric=metric)
grid = GridSearchCV(estimator=modelKnn, param_grid=param_grid, scoring='f1_macro', n_jobs=-1,cv=10)
grid.fit(X_train, Y_train) #70%

#Mejor modelo
modelKnn= grid.best_estimator_

#Medida de evaluación del mejor modelo
medidasH['Knn']=grid.best_score_

# Mejores párametros
print( grid.best_params_)
print(medidasH)
print("---- Tree Evaluacion 30-------")
#Evaluación del mejor Tree con el 30%
from sklearn import metrics

Y_pred = modelTree.predict(X_test)

#Matriz de confusion
metrics.plot_confusion_matrix(estimator=modelTree, X=X_test, y_true=Y_test, display_labels=labelencoder.classes_,values_format='d')  
plt.show()
#Precision, Recall, f1, exactitud
print(metrics.classification_report( y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
# Curva ROC
#plt.show() 
print("---- KNN Evaluacion 30-------")
#Evaluación de Knn
from sklearn import metrics

Y_pred = modelKnn.predict(X_test)

#Matriz de confusion
metrics.plot_confusion_matrix(estimator=modelKnn, X=X_test, y_true=Y_test, display_labels=labelencoder.classes_,values_format='d')  
plt.show()
#Precision, Recall, f1, exactitud
print(metrics.classification_report( y_true=Y_test, y_pred=Y_pred, target_names=labelencoder.classes_))
#plt.show() 
print("--------Exportando Modelo ----")
filename = 'modelo-clas.pkl'
variables= X.columns._values
pickle.dump([modelTree,labelencoder,variables,min_max_scaler], open(filename, 'wb'))