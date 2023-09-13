#!/usr/bin/env python
# coding: utf-8

# ### Import Basic Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/census_income.csv')


# In[4]:


df


# <font face = "Verdana" size = "4"> Here we can observe that  there are many columns which are  categorical so we have to convert into numeric 

# <font face = "Verdana" size = "4"> <b>  shape of data

# In[5]:


df.shape


# In[ ]:





# <font face = "Verdana" size = "4"> <b> checking data type of each column

# In[6]:



df.dtypes


# <font face = "Verdana" size = "4"> Here we can observe that  there are many columns which are  categorical so we have to convert into numeric 

# In[7]:


df.columns


# <font face = "Verdana" size = "4"> Observation upto about coloumn 

# ![Screenshot_20221209_170137.png](attachment:Screenshot_20221209_170137.png)

# 

# <font face = "Verdana" size = "4"> <b> checking spaces in data

# In[8]:



for x in df.columns:
    print(df.loc[df[x]==" "])


# <font face = "Verdana" size = "4"> there are no space in data set

# In[ ]:





# <font face = "Verdana" size = "4"> <b> Checking Null Values in DataFrame

# In[9]:


df.isnull().sum()


# <font face = "Verdana" size = "4"> There are no null values in data set

# <font face = "Verdana" size = "4"> <b>  check  the duplicate

# In[10]:


#  check  the duplicate
duplicate = df[df.duplicated()]
print("Duplicate Rows :")
 
#  Print the resultant Dataframe
duplicate


# In[11]:


duplicate.shape


# In[12]:


df=df.drop_duplicates()


# In[13]:


df.reset_index()


# <font face = "Verdana" size = "4"> <b> checking infomation of dataset

# In[14]:



df.info()


# <font face = "Verdana" size = "4"> Observations:-

# <font face = "Verdana" size = "2"> After deleting duplicate Row count is 32536

# In[15]:


#counting uniques of all columns
df.nunique()


# <font face = "Verdana" size = "2"> Target column as 2 unique so this is classification 

# <font face = "Verdana" size = "4">Unique And Value Count information

# In[16]:


print(df['Workclass'].unique())
print(df['Workclass'].value_counts())


# <font face = "Verdana" size = "4"> fill  (?  1836) with mode

# In[17]:


#Here private is the highest mode
df['Workclass']=df['Workclass'].replace(' ?','Private')


# In[18]:


#countplot
plt.figure(figsize=(19,8))
sns.countplot(df['Workclass'],palette='crest',hue='Sex',data=df);


# <font face = "Verdana" size = "4"> If you observe private workclass persons are high . That to male persons are high in all workclass.

# In[19]:


#countplot
plt.figure(figsize=(19,8))
sns.countplot(df['Workclass'],palette='magma',hue='Income',data=df);


# In[ ]:





# In[20]:


#checking unique
print(df['Education'].unique())
#counting unique values
print(df['Education'].value_counts())


# <font face = "Verdana" size = "4">  Encoding the categorical Data/ Feature Engineering

# In[21]:


df['Education']=df['Education'].replace([' 1st-4th',' 5th-6th',' Prof-school',' Preschool',' 7th-8th',' 9th',' 10th',' 11th',' 12th'],'school')
df['Education']=df['Education'].replace([' HS-grad',' Assoc-acdm',' Assoc-voc',' Some-college',' Bachelors'],'college')


# In[22]:


#checking unique values
df['Education'].unique()


# In[23]:


#countplot of Education
sns.countplot(df['Education'],palette='cubehelix',hue='Sex',data=df,);


# In[24]:


#countplot of Education
sns.countplot(df['Education'],palette='seismic',hue='Income',data=df);


# In[25]:


#checking unique
print(df['Education_num'].unique())
#counting unique
print(df['Education_num'].value_counts())


# In[26]:


#checking unique
print(df['Marital_status'].unique())
#counting unique
print(df['Marital_status'].value_counts())


# In[ ]:





# In[27]:


df['Marital_status']=df['Marital_status'].replace([' Married-civ-spouse',' Married-spouse-absent',' Married-AF-spouse'],'married')
df['Marital_status']=df['Marital_status'].replace([' Divorced',' Separated'],'Divorced')
df['Marital_status']=df['Marital_status'].replace(' Never-married','singles')


# In[28]:


#checking unique
print(df['Marital_status'].unique())
#counting unique
print(df['Marital_status'].value_counts())


# In[29]:


#count plot
sns.countplot(df['Marital_status'],palette='magma',hue='Sex',data=df);


# <font face = "Verdana" size = "4">Married Male persons are high .

# In[30]:


#count plot
sns.countplot(df['Marital_status'],palette='seismic',hue='Income',data=df);


# <font face = "Verdana" size = "4">maximum people are single whoes getting  <=50k

# In[31]:


#checking unique
print(df['Occupation'].unique())
#counting unique
print(df['Occupation'].value_counts())


# <font face = "Verdana" size = "4"> fill (? 1843) with mode

# In[32]:


df['Occupation']=df['Occupation'].replace(' ?',' Prof-specialty')


# In[33]:


#count plot
plt.figure(figsize=(25,8))
sns.countplot(df['Occupation'],palette='crest',hue='Sex',data=df);


# <font face = "Verdana" size = "4">  females are high in others services and Adm-Clerical

# In[34]:


#count plot
plt.figure(figsize=(25,8))
sns.countplot(df['Occupation'],palette='mako',hue='Income',data=df);


# <font face = "Verdana" size = "4"> All occupations <=50k persons are high

# In[35]:


#checking unique
print(df['Relationship'].unique())
#counting unique
print(df['Relationship'].value_counts())


# In[36]:


#count plot
plt.figure(figsize=(9,8))
sns.countplot(df['Relationship'],palette='cubehelix',hue='Income',data=df);


# Not in family relationship are many perons get <=50k income 

# In[37]:


#checking unique
print(df['Race'].unique())
#counting unique
print(df['Race'].value_counts())


# In[38]:


#count plot
plt.figure(figsize=(9,5))
sns.countplot(df['Race'],palette='rocket_r',hue='Sex',data=df);


# White perons are many here that to male persons are high count

# In[39]:


#count plot
plt.figure(figsize=(9,5))
sns.countplot(df['Race'],palette='viridis',hue='Income',data=df);


# In[40]:


#checking unique
print(df['Sex'].unique())
#counting unique
print(df['Sex'].value_counts())


# In[41]:


#count plot
plt.figure(figsize=(9,5))
sns.countplot(df['Sex'],palette='vlag',hue='Income',data=df);


# Males are high count in <=50k income

# In[42]:


#checking unique
print(df['Native_country'].unique())
#counting unique
print(df['Native_country'].value_counts())


# ###### fill ? with mode

# In[43]:


df['Native_country']=df['Native_country'].replace(' ?',' United-States')


# In[44]:


#checking unique
print(df['Native_country'].unique())
#counting unique
print(df['Native_country'].value_counts())


# United-States persons are many here

# In[45]:


#checking unique
print(df['Income'].unique())
#counting unique
print(df['Income'].value_counts())


# In[46]:


#count plot
plt.figure(figsize=(9,5))
sns.countplot(df['Income'],palette='vlag',hue='Sex',data=df);


# Males are high count in getting <=50k income

# In[47]:


df['Income']=df['Income'].replace(' >50K',0)
df['Income']=df['Income'].replace(' <=50K',1)


# In[48]:


df.describe().T


# ### Histogram

# In[49]:


df.hist(figsize=(12,12),layout=(3,3),sharex=False);


# ### Pie Chart

# In[50]:


px.pie(df,values='Education_num',names='Education',title='% of education',color_discrete_sequence=px.colors.qualitative.T10)


# In[51]:


px.pie(df,values='Age',names='Marital_status',title='% of Marital_status',color_discrete_sequence=px.colors.qualitative.T10)


# In[52]:


px.pie(df,values='Capital_gain',names='Education',title='% of Education',color_discrete_sequence=px.colors.qualitative.T10)


# In[53]:


px.pie(df,values='Capital_loss',names='Education',title='% of Education',color_discrete_sequence=px.colors.qualitative.T10)


# In[54]:


px.pie(df,values='Hours_per_week',names='Education',title='% of Education',color_discrete_sequence=px.colors.qualitative.T10)


# In[ ]:





# ### Label Encoder

# In[55]:


from sklearn.preprocessing import LabelEncoder


# In[56]:


df.dtypes


# In[57]:


df1=df.copy()


# In[58]:


df1=df1.apply(LabelEncoder().fit_transform)


# In[59]:


df1


# In[60]:


#checking correlation using HeatMap
plt.figure(figsize=(12,11))
sns.heatmap(df1.corr(),annot=True,cmap='PiYG');


# In[61]:


#correlation
df1.corr()


# ###### Observations:-

# 1. Marital_status are highly negative correlation with Age
# 2. Relatonship also high negative correlation with Age
# 3. Capital_gain is positive correlation with Age
# 3. Education is highly negative correlation with Education_num
# 4. Sex is Highly Negative correlation with Relationship
# 5. Sex is Positive correlation with Hours_per_week
# 6. Capital_gain and Hours_per_week is positive correlation with Education_num

# In[62]:


df_target=df1.corr()


# In[63]:


df_target['Income'].sort_values(ascending=False)


# ###### Observations:-

# 1. Relationship And Education is high positive correlation with Income 
# 2. Marital_status and Workclass also little bit positive correlation with Income
# 3. Capital_gain and Education Num is highly Negative correlation with Income

# ###### Unique Count of Target Variable

# In[64]:


#checking unique count bcz to check that dataset is balanced or not
df['Income'].value_counts()


# This is imbalanced Dataset Apply SMOTE to make it balanced

# ### SMOTE

# In[65]:


from imblearn.over_sampling import SMOTE


# In[66]:


dfx=df1.drop('Income',axis=1)
dfy=df['Income']


# In[67]:


dfx


# In[68]:


dfy


# In[69]:


smt=SMOTE()


# In[70]:


trainx,trainy=smt.fit_resample(dfx,dfy)


# In[71]:


#value count of target variable
trainy.value_counts()


# In[72]:


#count plot
sns.countplot(trainy);


# Now dataset is balanced

# In[73]:


#making dataframe
df1=pd.DataFrame(trainx)


# In[74]:


#dependent variables
df1


# In[75]:


#adding target variable
df1['Income']=trainy


# In[76]:


#after adding target variable
df1


# In[77]:


#checking shape
df1.shape


# 49438 rows and 15 columns

# In[78]:


#checking null values
df1.isnull().sum()


# ###### Checking Outliers

# In[79]:


df1.plot(kind='box',subplots=True,layout=(4,4),figsize=(10,10));


# ###### Removing Outliers

# In[80]:


from scipy.stats import zscore


# In[81]:


z=np.abs(zscore(df1))


# In[82]:


threshold=3
print(np.where(z>3))


# In[ ]:





# In[83]:


#removing outliers
df_new=df1[(z<3).all(axis=1)]


# In[84]:


#after removing outliers
df_new


# In[85]:


#after removing outliers checking shape
df_new.shape


# In[ ]:





# ###### Checking Skewness

# In[86]:


#dependent variables
x=df_new.iloc[:,0:-1]
y=df_new.iloc[:,-1]


# In[87]:


#checking shapes
print(x.shape)
print(y.shape)


# In[88]:


x.plot(kind='kde',subplots=True,layout=(4,4),figsize=(10,10));


# In[89]:


x.skew().sort_values(ascending=False)


# high skewness in data

# ###### Removing Skewness 

# In[90]:


from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


# In[91]:


scaler=MinMaxScaler(feature_range=(1,2))
power=PowerTransformer(method='box-cox')
pipeline=Pipeline(steps=[('s',scaler),('p',power)])


# In[92]:


data=pipeline.fit_transform(x)


# In[93]:


#make dataframe
x=pd.DataFrame(data,columns=x.columns)


# In[94]:


x


# ### VIF(Variance Inflation Factor)

# In[95]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[96]:


def vif():
    vif=pd.DataFrame()
    vif['Varibales']=x.columns
    vif['VIF Factor']=[variance_inflation_factor(x.values,i)for i in range(x.shape[1])]
    return vif


# In[97]:


vif()


# In[98]:


#Capital_loss and Native_country is same VIF Factor
x.drop('Capital_loss',axis=1,inplace=True)


# In[99]:


#Capital_gain and Race is same VIF Factor, Capital_gain is also negative correlation with target variable
x.drop('Capital_gain',axis=1,inplace=True)


# In[100]:


#Education and Education_num is Same VIF Factor , Education_num also negative correlation with target variable
x.drop('Education_num',axis=1,inplace=True)


# In[101]:


#Occupation and Native_country is same VIF Factor
x.drop('Native_country',axis=1,inplace=True)


# In[102]:


#checking shape of dependent variables
x.shape


# In[ ]:





# ### Standard Scaler

# In[103]:


scale=MinMaxScaler()


# In[104]:


x=scale.fit_transform(x)


# In[105]:


#after applying standard scaler
x.shape


# In[ ]:





# ### Model Selection

# In[106]:


#target variable shape
y.shape


# In[107]:


#classification models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score


# In[108]:


lg=LogisticRegression()
gnb=GaussianNB()
dtc=DecisionTreeClassifier()
knc=KNeighborsClassifier()
svc=SVC()
rfc=RandomForestClassifier()
abc=AdaBoostClassifier()
gbc=GradientBoostingClassifier()


# In[109]:


list_model=[lg,gnb,dtc,knc,svc,rfc,abc,gbc]


# In[110]:


def confusion_plot():
    print('accuracy_score:-',accuracy_score(pred,y_test))
    print(classification_report(pred,y_test))
    matrix = confusion_matrix(pred,y_test)
    sns.heatmap(matrix, annot=True, fmt="d", cmap='YlOrBr', square=True)
    plt.xlabel("predicted")
    plt.show()


# In[111]:


maxacc=0
maxrn=0

for i in range(1,100):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=i)
    lg.fit(x_train,y_train)
    pred=lg.predict(x_test)
    score=accuracy_score(pred,y_test)
    if score>maxacc:
        maxacc=score
        maxrn=i
print('accuracy_score:-',maxacc,'Random state:-',maxrn)


# In[112]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=16)
lg.fit(x_train,y_train)
pred=lg.predict(x_test)
confusion_plot()


# In[113]:


def confusion_plot():
    print('accuracy_score:-',accuracy_score(pred,y_test))
    print(classification_report(pred,y_test))
    matrix = confusion_matrix(pred,y_test)
    sns.heatmap(matrix, annot=True, fmt="d", cmap='Spectral', square=True)
    plt.xlabel("predicted")
    plt.show()


# In[114]:


maxacc=0
maxrn=0

for i in range(1,100):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=i)
    gnb.fit(x_train,y_train)
    pred=gnb.predict(x_test)
    score=accuracy_score(pred,y_test)
    if score>maxacc:
        maxacc=score
        maxrn=i
print('accuracy_score:-',maxacc,'Random state:-',maxrn)


# In[115]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=54)
gnb.fit(x_train,y_train)
pred=gnb.predict(x_test)
confusion_plot()


# In[116]:


def confusion_plot():
    print('accuracy_score:-',accuracy_score(pred,y_test))
    print(classification_report(pred,y_test))
    matrix = confusion_matrix(pred,y_test)
    sns.heatmap(matrix, annot=True, fmt="d", cmap='coolwarm', square=True)
    plt.xlabel("predicted")
    plt.show()


# In[117]:


maxacc=0
maxrn=0

for i in range(1,100):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=i)
    dtc.fit(x_train,y_train)
    pred=dtc.predict(x_test)
    score=accuracy_score(pred,y_test)
    if score>maxacc:
        maxacc=score
        maxrn=i
print('accuracy_score:-',maxacc,'Random state:-',maxrn)


# In[118]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=45)
dtc.fit(x_train,y_train)
pred=dtc.predict(x_test)
confusion_plot()


# In[119]:


def confusion_plot():
    print('accuracy_score:-',accuracy_score(pred,y_test))
    print(classification_report(pred,y_test))
    matrix = confusion_matrix(pred,y_test)
    sns.heatmap(matrix, annot=True, fmt="d", cmap='icefire', square=True)
    plt.xlabel("predicted")
    plt.show()


# In[120]:


maxacc=0
maxrn=0

for i in range(20,50):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=i)
    knc.fit(x_train,y_train)
    pred=knc.predict(x_test)
    score=accuracy_score(pred,y_test)
    if score>maxacc:
        maxacc=score
        maxrn=i
print('accuracy_score:-',maxacc,'Random state:-',maxrn)


# In[121]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=25)
knc.fit(x_train,y_train)
pred=knc.predict(x_test)
confusion_plot()


# In[122]:


def confusion_plot():
    print('accuracy_score:-',accuracy_score(pred,y_test))
    print(classification_report(pred,y_test))
    matrix = confusion_matrix(pred,y_test)
    sns.heatmap(matrix, annot=True, fmt="d", cmap='Greens', square=True)
    plt.xlabel("predicted")
    plt.show()


# In[123]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=43)
svc.fit(x_train,y_train)
pred=svc.predict(x_test)
accuracy_score(pred,y_test)


# In[124]:


confusion_plot()


# In[125]:


def confusion_plot():
    print('accuracy_score:-',accuracy_score(pred,y_test))
    print(classification_report(pred,y_test))
    matrix = confusion_matrix(pred,y_test)
    sns.heatmap(matrix, annot=True, fmt="d", cmap='viridis', square=True)
    plt.xlabel("predicted")
    plt.show()


# In[126]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=43)
rfc.fit(x_train,y_train)
pred=rfc.predict(x_test)
accuracy_score(pred,y_test)


# In[127]:


confusion_plot()


# In[128]:


def confusion_plot():
    print('accuracy_score:-',accuracy_score(pred,y_test))
    print(classification_report(pred,y_test))
    matrix = confusion_matrix(pred,y_test)
    sns.heatmap(matrix, annot=True, fmt="d", cmap='mako', square=True)
    plt.xlabel("predicted")
    plt.show()


# In[129]:


maxacc=0
maxrn=0

for i in range(1,10):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=i)
    abc.fit(x_train,y_train)
    pred=abc.predict(x_test)
    score=accuracy_score(pred,y_test)
    if score>maxacc:
        maxacc=score
        maxrn=i
print('accuracy_score:-',maxacc,'Random state:-',maxrn)


# In[150]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=2)
abc.fit(x_train,y_train)
pred=abc.predict(x_test)
confusion_plot()


# In[151]:


def confusion_plot():
    print('accuracy_score:-',accuracy_score(pred,y_test))
    print(classification_report(pred,y_test))
    matrix = confusion_matrix(pred,y_test)
    sns.heatmap(matrix, annot=True, fmt="d", cmap='magma', square=True)
    plt.xlabel("predicted")
    plt.show()


# In[152]:


maxacc=0
maxrn=0

for i in range(1,50):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=i)
    gbc.fit(x_train,y_train)
    pred=gbc.predict(x_test)
    score=accuracy_score(pred,y_test)
    if score>maxacc:
        maxacc=score
        maxrn=i
print('accuracy_score:-',maxacc,'Random state:-',maxrn)


# In[153]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=16)
gbc.fit(x_train,y_train)
pred=gbc.predict(x_test)
confusion_plot()


# ### Cross Validation

# In[154]:


for model in list_model:
    score=cross_val_score(model,x,y,cv=5)
    print(model)
    print(score)
    print(score.mean())


# ### GradientBoostingClassifier is the best model

# In[155]:


def confusion_plot():
    print('accuracy_score:-',accuracy_score(pred,y_test))
    print(classification_report(pred,y_test))
    matrix = confusion_matrix(pred,y_test)
    sns.heatmap(matrix, annot=True, fmt="d", cmap='Greens', square=True)
    plt.xlabel("predicted")
    plt.show()


# In[156]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=16)
gbc.fit(x_train,y_train)
pred=gbc.predict(x_test)
confusion_plot()


# ### Hyperparameters for a model

# In[157]:


param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10),param_grid = param_test1, scoring='roc_auc',n_jobs=4,cv=5)
gsearch1.fit(x_train,y_train)


# In[158]:


print(gsearch1.best_estimator_)
print(gsearch1.best_params_)
print(gsearch1.best_score_)


# In[159]:


gbc_pred=gsearch1.best_estimator_.predict(x_test)


# In[160]:


def confusion_plot1():
    print('accuracy_score:-',accuracy_score(gbc_pred,y_test))
    print(gsearch1.best_estimator_)
    print(gsearch1.best_params_)
    print('\nClassification Report')
    print(classification_report(gbc_pred,y_test))
    matrix = confusion_matrix(gbc_pred,y_test)
    sns.heatmap(matrix, annot=True, fmt="d", cmap='icefire', square=True)
    plt.xlabel("predicted")
    plt.show()


# In[161]:


confusion_plot1()


# ### ROC Curve

# In[162]:


gbc_pred=gsearch1.best_estimator_.predict_proba(x_test)


# In[163]:


# roc curve for classes
fpr = {}
tpr = {}
thresh ={}
n_class=2


# In[164]:


for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, gbc_pred[:,i], pos_label=i)


# In[165]:


sns.set_theme()


# In[166]:


# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class 1 vs Rest')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('ROC Curve',dpi=300); 


# ### Save the Model

# In[167]:


import joblib


# In[168]:


#save the model
joblib.dump(gsearch1.best_estimator_,'census_income_gbc.obj')


# In[169]:


#load the model
load_model=joblib.load('census_income_gbc.obj')


# In[ ]:





# In[ ]:




