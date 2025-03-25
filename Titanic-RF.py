#!/usr/bin/env python
# coding: utf-8

# In[10]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

get_ipython().system('apt-get install git -y')
#installs git

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


dset_learn = pd.read_csv("/kaggle/input/titanic/train.csv")
dset_learn.head()


# In[3]:


test_data=pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# In[4]:


women = dset_learn.loc[dset_learn.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)


# In[5]:


QueenstownSurv = dset_learn.loc[dset_learn.Embarked == 'Q']["Survived"]
rate_QS = sum(QueenstownSurv)/len(QueenstownSurv)

print("% of Queenstown-embarked that lived:", rate_QS)


# In[6]:


from sklearn.ensemble import RandomForestClassifier

y = dset_learn["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(dset_learn[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[15]:


#get_ipython().system('git init')
#get_ipython().system('git config --global user.email "amaung0451@gmail.com"')
#get_ipython().system('git config --global user.name "AdamM"')
#get_ipython().system('git config --global init.defaultBranch main')
#get_ipython().system('git branch -m main')


# In[17]:


#get_ipython().system('git add titanic-ml.ipynb')
#get_ipython().system('git commit -m "Pushed from Kaggle"')
#get_ipython().system('git push -u origin main')

