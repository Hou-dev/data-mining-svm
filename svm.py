#!/usr/bin/env python
# coding: utf-8

# In[6]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm


# In[7]:


x=[1,5,1.5,8,1,9]
y=[2,8,1.8,8,0.6,11]


# In[8]:


plt.scatter(x,y)
plt.show()


# In[9]:


x = np.array([[1,2],
            [5,8],
            [1.5,1.8],
            [8,8],
            [1,0.6],
            [9,11]])
y=[0,1,0,1,0,1]
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(x,y)


# In[10]:


k = np.array([[0.58,0.7]])
k.reshape(-1,1)
print(clf.predict(k))


# In[11]:


k = np.array([[10.58,10.7]])
k.reshape(-1,1)
print(clf.predict(k))


# In[12]:


w = clf.coef_[0]
print(w)
a = -w[0] / w[1]
xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx,yy,'k-',label='SVM example')

plt.scatter(x[:,0],x[:,1], c =y)
plt.legend()
plt.show()


# In[13]:


import pandas as pd
bankdata = pd.read_csv('bill_authentication.csv')


# In[14]:


bankdata.shape


# In[15]:


bankdata.head()


# In[16]:


x = bankdata.drop('Class', axis =1)
y = bankdata['Class']


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.20)


# In[18]:


from sklearn.svm import SVC
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(x_train,y_train)


# In[19]:


y_pred = svclassifier.predict(x_test)


# In[20]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[21]:


from sklearn import datasets
iris = datasets.load_iris()
print(list(iris))
print(iris.feature_names)


# In[26]:


x,y =iris.data, iris.target
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2).fit(x)
labels = gmm.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=labels, s=40, cmap='viridis');


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.20)


# In[28]:


svclassifier = SVC(kernel = 'linear', C =100)
svclassifier.fit(x_train,y_train)
y_pred = svclassifier.predict(x_test)


# In[29]:


plot = plt.scatter(y_test,y_pred)


# In[30]:


from sklearn.metrics import roc_auc_score

print(confusion_matrix(y_test,y_pred))
#print(roc_auc_score(y-test,y_pred))


# In[31]:


from sklearn.model_selection import RepeatedKFold
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
for train, test in rkf.split(x):
    print("%s %s" % (train, test))


# In[32]:


from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train, test in loo.split(x):
    print("%s %s" % (train, test))


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
svclassifier = SVC(kernel = 'rbf', C =1)
svclassifier.fit(x_train,y_train)
y_pred = svclassifier.predict(x_test)
plot = plt.scatter(y_test,y_pred)

