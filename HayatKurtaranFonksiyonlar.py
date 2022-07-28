#!/usr/bin/env python
# coding: utf-8

# #Kütüphaneler

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


# #Dataframe'i Kontrol için :

# In[ ]:


def check_df(dataframe, head=8):
  print("##### Shape #####")
  print(dataframe.shape)
  print("##### Types #####")
  print(dataframe.dtypes)
  print("##### Tail #####")
  print(dataframe.tail(head))
  print("##### Head #####")
  print(dataframe.head(head))
  print("##### Null Analysis #####")
  print(dataframe.isnull().sum())
  print("##### Quantiles #####")
  print(dataframe.describe([0,0.05, 0.50, 0.95, 0.99, 1]).T)


# # Cat_summary :

# In[ ]:





# In[ ]:


def cat_summary(dataframe, col_name, plot = False, hue = False):
   print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                      "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
   print("***********************************************************")
   
   if plot:
    if hue:
      sns.countplot(x=dataframe[col_name], hue = col_name, data = dataframe)
      plt.show(block=True)
    else:
      sns.countplot(x=dataframe[col_name], data = dataframe)
      plt.show(block=True)


# # Nume_summary:

# In[ ]:


def num_summary(dataframe, num_col, plot = False):
  quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
  print(dataframe[num_col].describe(quantiles).T)

  if plot:
    dataframe[num_col].hist()
    plt.xlabel(num_col)
    plt.title(num_col)
    plt.show(block = True)

  print("\n\n")


# #Hedef degiskenin kategorik degiskenler ile analizi için:

# In[ ]:


def target_sum_with_cat(dataframe, target, categorical_col):
  print(pd.DataFrame({"TARGET_MEAN" :dataframe.groupby(categorical_col)[target].mean()}))


# #Hedef degiskenin nümerik degiskenler ile analizi için:

# In[ ]:


def target_sum_with_num(dataframe, target, numerical_col):
  print(dataframe.groupby(target).agg({numerical_col:"mean"}))


# #Kolonları çeşitlerine (numeric, categoric, cardinal) ayırmak için :
# 

# In[ ]:


def grab_col_names(dataframe, cat_th=10, car_th=20): #essiz deger sayisi 10dan kucukse kategorik degisken, 20 den buyukse de kardinal degisken gibi dusunucez. 
  cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

  num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int64","float64"]]
  
  cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

  cat_cols = num_but_cat + cat_cols
  cat_cols = [col for col in cat_cols if col not in cat_but_car]

  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64","float64"]]
  num_cols = [col for col in num_cols if col not in cat_cols]

  print(f"Observations: {dataframe.shape[0]}")
  print(f"Variables: {dataframe.shape[1]}")
  print(f"Categorical Columns: {len(cat_cols)}")
  print(f"Numerical Columns: {len(num_cols)}")
  print(f"Categoric but Cardinal: {len(cat_but_car)}")
  print(f"Numeric but Categoric: {len(num_but_cat)}")

  return cat_cols, num_cols, cat_but_car


# #Yuksek Korelasyonlu Degiskenlerin listesi için :
# 

# In[ ]:


def high_correlated_cols(dataframe, plot= False, corr_th = 0.90):
  corr = dataframe.corr()
  cor_matrix = corr.abs()
  upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
  drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
  if plot:
    sns.set(rc={'figure.figsize': (15,15)})
    sns.heatmap(corr, cmap="RdBu")
    plt.show()
  return drop_list


# #Aykırı değerimizi saptama işlemi için:
# 

# In[ ]:


def outlier_thresholds(dataframe, col_name, q1 = 0.25, q3 = 0.75):
  quantile1 = dataframe[col_name].quantile(q1)
  quantile3 = dataframe[col_name].quantile(q3)
  interquantile_range = quantile3 - quantile1
  up_limit = quantile3 + 1.5 * interquantile_range
  low_limit = quantile1 - 1.5 * interquantile_range
  return low_limit, up_limit


# #Thresholdlara göre outlier var mı yok mu diye kontrol etmek için:
# 

# In[ ]:


def check_outlier(dataframe, col_name):
  low_limit, up_limit = outlier_thresholds(dataframe, col_name)
  if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
    return True
  else:
    return False


# #Var olan outlierları görmek için:
# 

# In[ ]:


def grab_outliers(dataframe, col_name, index=False):
  low, up = outlier_thresholds(dataframe, col_name)
  if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
    print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
  else:
    print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

  if index:
    outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
    return outlier_index


# #Aykırı değerleri silme işlemi gerçekleştirmek için :
# 

# In[ ]:


def remove_outliers(dataframe, col_name, index=False):
  low_limit, up_limit = outlier_thresholds(dataframe, col_name)
  df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
  return df_without_outliers


# #Baskılama yöntemi uygulayarak, aykırı değerleri low ve uplarla değiştirmek için :
# 

# In[ ]:


#Atama yapmamıza gerek yok çünkü fonksiyonun içerisinde kullandığımız loc yapısından dolayı kalıcı değişiklik yapıyor olacak.
def replace_with_thresholds(dataframe, variable):
  low_limit, up_limit = outlier_thresholds(dataframe, variable)
  dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
  dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# #Eksikliğe sahip verilerin seçilmesi için:

# In[ ]:


def missing_values_table(dataframe, na_name=False): #na_name eksik değerlerin bullunduğu değişkenlerin ismi
  na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
  n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
  ratio = (dataframe[na_columns].isnull().sum()/dataframe.shape[0] *100).sort_values(ascending=False)
  missing_df=pd.concat([n_miss, np.round(ratio,2)], axis=1, keys=['n_miss','ratio']) #concat ile birleştiriyoruz.
  print(missing_df, end="\n")
  if na_name:
    return na_columns


# # Eksik değerlerin bağımlı değişken ile analizi için:
# 

# In[ ]:


#Tahmine dayalı atama ile doldurma
def missing_vs_target(dataframe, target, na_columns):
  temp_df = dataframe.copy() #dataframe in bir kopyasını oluşturduk. İstersek oluşturmayadabiliriz. Orjinal veriler elinizde kalsın istiyorsanız.
  for col in na_columns: #boş kolonlar arasında geziyoruz.
    temp_df[col +"_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0) # İlgili kolonların boş değerler barındırdığını belirtmek amacıyla sonuna _NA_FLAG ekleyerek işaretledik. Bu değerlere kolonda boş değerlerin yerine 1, dolu değerlerin yerine 0 atadık.
  
  na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns #Flagli kolonları; loc kullanarak bütün sütunları al, kolonlardan da içerisinde "_NA_" bulunanları al diyerek na_flags içerisine atadık.
  for col in na_flags:#na bulunan kolonları gez.
    print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(), #Bu kolonlara göre grupby a alarak target ın ortalamasını yaz.  
                        "Count" : temp_df.groupby(col)[target].count()}), end="\n\n\n") #Bu kolonlara göre grupby a alarak target ın countunu yaz.


# # Label encoder işlemi için:

# In[ ]:


def label_encoder(dataframe, binary_col):
  labelencoder=LabelEncoder()
  dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
  return dataframe


# ## One Hot Encoder için :

# In[ ]:


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
  dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
  return dataframe


# # Rare Analyser -> Elimizde bol kategorik değişkenli bir veri seti olduğunda çok işe yarar ve kullanılır.

# In[ ]:


def rare_analyser(dataframe, target, cat_cols):
  for col in cat_cols:
    print(col,":", len(dataframe[col].value_counts()))#ilgili kategorik değişkenin kaç sınıfı olduğu bilgisi.
    print(pd.DataFrame({"COUNT":dataframe[col].value_counts(),#sınıf frekansları
                        "RATIO":dataframe[col].value_counts()/len(dataframe),#sınıf oranları
                        "TARGET_MEAN":dataframe.groupby(col)[target].mean()}), end="\n\n\n") #target yani bağımlı değişkene göre groupby işlemi.


# #Label Encoder işlemi için:
# 

# In[ ]:


def label_encoder(dataframe, binary_col):
  labelencoder=LabelEncoder()
  dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
  return dataframe


# #One Hot Encoder işlemi için:

# In[ ]:


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
  dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
  return dataframe


# #Rare Analiz işlemi için:

# In[ ]:


def rare_analyser(dataframe, target, cat_cols):
  for col in cat_cols:
    print(col,":", len(dataframe[col].value_counts()))#ilgili kategorik değişkenin kaç sınıfı olduğu bilgisi.
    print(pd.DataFrame({"COUNT":dataframe[col].value_counts(),#sınıf frekansları
                        "RATIO":dataframe[col].value_counts()/len(dataframe),#sınıf oranları
                        "TARGET_MEAN":dataframe.groupby(col)[target].mean()}), end="\n\n\n")


# #Rare Encoder işlemi için:

# In[ ]:


def rare_encoder(dataframe, rare_perc):
  temp_df = dataframe.copy()
  #eğer fonksiyona girilen rare oranından daha düşük sayıda herhangi bir, bu kategorik değişkenin sınıf oranı varsa aynı zamanda bu bir kategorik değişkense bunları rare columnları olarak getir.
  rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O" and (temp_df[col].value_counts()/len(temp_df)<rare_perc).any(axis=None)]

  for var in rare_columns:
    tmp = temp_df[var].value_counts()/ len(temp_df) #alınan rare columlarının değerlerini toplam gözlem sayısına oranını alıyoruz. İlgili rare değişkeni için sınıf oranları hesaplandı.
    rare_labels = tmp[tmp<rare_perc].index #Çalışmanın başında verilen orandan daha düşük orana sahip olan sınıflarla veri setine indirge ve bunları rare_labelde tut. 
    temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var]) #eğer rare columnlardan birinde eğer rare label olma durumunda bunların yerine "Rare" yaz. Değilse olduğu gibi kalacak.
    #Pandas isin()yöntemi, veri çerçevelerini filtrelemek için kullanılır. isin() yöntem, belirli bir sütunda belirli bir (veya Çoklu) değere sahip satırların seçilmesine yardımcı olur.
  return temp_df


# # İki grubun oranını hesaplamak için :

# In[ ]:


from statsmodels.stats.proportion import proportions_ztest
test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_CABIN_BOOL"] == 1, ("Survived")].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
                                      nobs = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])


# #Yeni türettiğimiz değişkenlere bakalım istersek:
# 

# In[ ]:


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
def plot_importance(model,features, num=len(X),save=False):
  feature_imp = pd.DataFrame({"Value" : model.feature_importances_, "Feature" : features.columns})
  plt.figure(figsize=(10,10))
  sns.set(font_scale=1)
  sns.barplot(x="Value", y = "Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
  plt.title("Features")
  plt.tight_layout()
  plt.show()
  if save:
    plt.savefig("importances.png")

