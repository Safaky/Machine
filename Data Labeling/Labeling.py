# %%
import numpy as np
import pandas as pd
  
# Import dataset
df = pd.read_csv('montgomery_metadata-3.csv')
df.sample(10)

# %%
category=pd.cut(df.age, bins=[0,2,17,65,99], labels=['Todler/baby', 'Child', 'Adult', 'Elderly'])
df.insert(4,'Age Group', category)
df.sample(10)

#######################################################################################################

#If you want a binary age classification :

#category=pd.cut(df.age, bins=[0,17,99], labels=[ 'Child', 'Adult'])
#df.insert(4,'Age Group', category)
#df.sample(10)
#categories=pd.Categorical(df['Age Group'],categories=['Adult','Child'],ordered=True)
#AgeGroup,unique=pd.factorize(categories,sort=True)
#df['Binary Age']=AgeGroup
#df['Binary Age']
#df.sample(10)

#######################################################################################################

# %%
categories=pd.Categorical(df['findings'],categories=['normal','Sick'],ordered=True)
findings,unique=pd.factorize(categories,sort=True)
df['findings']=findings
df['findings']
df.sample(10)

# %%
categories=pd.Categorical(df['gender'],categories=['Male','Female'],ordered=True)
gender,unique=pd.factorize(categories,sort=True)
df['gender']=gender
df['gender']
df.sample(10)

# %%
from sklearn import preprocessing
  
    
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
    
df['study_id']= label_encoder.fit_transform(df['study_id'])
  
df['study_id'].unique()

# %%
df.sample(10)

# %%
df.to_csv("output.csv")

# %%
