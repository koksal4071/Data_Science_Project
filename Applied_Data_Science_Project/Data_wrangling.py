# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:46:20 2024

@author: Abdk
"""

import pandas as pd
import numpy as np

df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
df.head(10)

df.isnull().sum()/df.count()*100
df.dtypes
df["LaunchSite"].value_counts()
df["Orbit"].value_counts()

landing_outcomes = df["Outcome"].value_counts()
landing_outcomes

for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)


bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes

def onehot(item):
    if item in bad_outcomes:
        return 0
    else:
        return 1
landing_class = df["Outcome"].apply(onehot)
landing_class

df['Class']=landing_class
df[['Class']].head(8)
df.head(5)
df["Class"].mean()


