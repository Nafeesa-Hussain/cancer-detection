#!/usr/bin/env python
# coding: utf-8

# In[54]:


import os


# In[55]:


os.getcwd()


# In[56]:


os.chdir("C:/Users/Nafeesa Hussain/Desktop")


# In[57]:


os.getcwd()


# In[58]:


import numpy as np


# In[59]:


import pandas as pd


# In[60]:


from sklearn import preprocessing


# In[61]:


data1=pd.read_csv("cancer.csv")


# In[62]:


data1


# In[63]:


data1=data1.drop(["id"],axis=1)


# In[64]:


data1


# In[65]:


data1.diagnosis.value_counts()


# In[66]:


data1['diagnosis'].replace(['M','B'],['Malignant','Benign'],inplace=True)


# In[67]:


data1


# In[68]:


import matplotlib.pyplot as plt


# In[69]:


x=data1.loc[:,['radius_mean','texture_mean','perimeter_mean']]


# In[70]:


x


# In[71]:


x.describe()


# In[72]:


x=data1.iloc[:,1:31]
y=data1.iloc[:,0]


# In[73]:


x


# In[74]:


y


# In[75]:


data1['diagnosis'].replace(['Malignant','Benign'],['0','1'],inplace=True)


# In[76]:


data1


# In[77]:


minmax=preprocessing.MinMaxScaler(feature_range=(0,1))
minmax.fit(x).transform(x)


# In[78]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[79]:


x_train


# In[80]:


y_train


# In[81]:


x_test


# In[82]:


y_test


# In[83]:


y_train=y_train.astype('int')
y_train


# In[84]:


y_test=y_test.astype('int')
y_test


# In[85]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)


# In[86]:


y_pred=classifier.predict(x_test)


# In[87]:


y_pred


# In[88]:


print("Actual breast cancer")
print(y_test.values)


# In[89]:


from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix


# In[90]:


print("\n Accuracy score:%f"%(accuracy_score(y_test,y_pred)*100))
print("\n Recall score:%f"%(recall_score(y_test,y_pred)*100))
print("\n ROC score:%f"%(roc_auc_score(y_test,y_pred)*100))
print(confusion_matrix(y_test,y_pred))


# In[91]:


probas=classifier.predict_proba(x_test)


# In[92]:


probas


# In[93]:


#dixal prob image
plt.figure(dpi=1500)
plt.hist(probas,bins=20)
plt.title('Classification Probabilities')
plt.xlabel('Probability')
plt.ylabel('# of Instances')
plt.xlim([0.5,1.0])
plt.legend(y_test)
plt.show()


# In[ ]:





# In[94]:


x_train_std=minmax.fit_transform(x_train)
x_test_std=minmax.transform(x_test)


# In[95]:


#validation
from sklearn.model_selection import cross_val_score,cross_val_predict


# In[96]:


classifier_acc=cross_val_score(classifier,x_train_std,y_train,cv=3,scoring="accuracy",n_jobs=-1)


# In[97]:


classifier_acc


# In[98]:


classifier_proba=cross_val_predict(classifier,x_train_std,y_train,cv=3,method='predict_proba')


# In[99]:


classifier_scores=classifier_proba[:,1]


# In[100]:


classifier_scores


# In[101]:


#fpr-false pos rate tpr-true pos rate
from sklearn.metrics import roc_auc_score,roc_curve 
def Roc_curve(title,y_train,scores,label=None):
    #calculate the ROC score
    fpr,tpr,thresholds=roc_curve(y_train,scores)
    print('AUC Score({}):{:2f}'.format(title,roc_auc_score(y_train,scores)))
    plt.figure(figsize=(8,6))
    plt.plot(fpr,tpr,linewidth=2,label=label,color='b')
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.title('ROC Curve:{}'.format(title),fontsize=16)
    plt.show()


# In[102]:


Roc_curve('kNN',y_train,classifier_scores)


# In[ ]:




