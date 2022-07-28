import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from HayatKurtaranFonksiyonlar import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("Datasets/2020_01/2020_01.csv", encoding='unicode_escape')
df.head()
df.info()

df_processed3 = pd.read_csv("Datasets/processed_2020_03/output/processed_2020_03.csv")
df.head()
df_processed3.info()
df_processed3.describe()
df_processed3.shape   # 26 columns

cat_cols, num_cols, cat_but_car = grab_col_names(df_processed3)

len(cat_cols)
len(num_cols) 
len(cat_but_car)  #these are useless columns but do not drop them yet


df_processed3["AckReq"].nunique()

#############################################################################################################################
#categorical variable summary
#############################################################################################################################

for col in cat_cols:
    cat_summary(df_processed3, col, plot=False)
#notes:
"""
-Categorical columns have less than 10 unique values. 

Questions: 
-"Severity": what does 500 and 900 represent?
-"Quality": What does 0, 64 and 192 represent?
-What is "Mask"? What does its values represent?
-What is "Newstate"? What does its values represent?
-What does 0 and 1 in "Status" represent?
-"AckReq" and "Area"


-"Cookie", "Shelving", "MachineName", "ServerProgID", "ServerNodeName", "SubscriptionName", "EventType", "EventCategory" has only 1 unique 
value. So these columns are not useful and should be removed.

-"ActorID", "Priority" has no values. I guess all of them are nans. They should be removed.

-Some unique values have very low percentage such as the last datas in "Area" feature.

-False rate in "AckReq" is less than 1%

-"5" and "7" are less than 1% in "Newstate" feature.

-Last 4 unique variables in "Mask" have less than 1%.

-"64" is 0.003% in "Quality" feature.

"""

#drop the following columns (empty columns)
df_processed3.drop(["ActorID", "Priority"], axis=1, inplace=True)
df_processed3.shape

#drop the following columns (only 1 unique value)
df_processed3.drop(["Cookie", "Shelving", "MachineName", "ServerProgID", "ServerNodeName", "SubscriptionName", "EventType", "EventCategory"], axis=1, inplace=True)
df_processed3.shape

#############################################################################################################################
#numerical variable summary
#############################################################################################################################
num_summary(df_processed3, num_cols,plot=True)
df_processed3[num_cols].head()

"""
"EventTimeMS" and "ActiveTimeMS" should be explained. 
"""

# All the values represent an alarm occurance. So, what will the target variable be?
df_processed3["Target"] = 1  # 1 for alarm occurance
for col in cat_cols:
    target_sum_with_cat(df_processed3, "Target", col)  #all of them are 1 bc there is no non alarm occurance in dataset


