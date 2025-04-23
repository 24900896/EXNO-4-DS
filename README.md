EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
  ~~~  
    import pandas as pd
    from scipy import stats
    import numpy as np
    df=pd.read_csv("/content/bmi.csv")
    df.head()
 ~~~   
![image](https://github.com/user-attachments/assets/d5a268af-b1f4-4741-a3a4-4dab949ffb04)
~~~
df.dropna()
~~~
![image](https://github.com/user-attachments/assets/331667bc-a326-4c0b-9adf-dc916d897966)
~~~
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
~~~
![image](https://github.com/user-attachments/assets/0fe049f1-380f-48d2-9e21-cace27e0aa21)
~~~
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
~~~
![image](https://github.com/user-attachments/assets/4f343fe5-a992-4a80-8afd-0f9551d0dea6)
~~~
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
~~~
![image](https://github.com/user-attachments/assets/992b611e-de5f-4bc6-9171-5538ccb57e88)
~~~
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
~~~
![image](https://github.com/user-attachments/assets/5668adb0-5d4e-4bfd-902f-1e346bcfba29)
~~~
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
~~~
![image](https://github.com/user-attachments/assets/06996cdc-be2c-4e2b-9563-fac5d8cfb24c)
~~~
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
~~~
![image](https://github.com/user-attachments/assets/06580ef5-cc95-4d45-ab72-7c08aefb0d00)
~~~
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
~~~
![image](https://github.com/user-attachments/assets/8e4096b1-a458-4b13-b19e-40a18fec93d8)
~~~
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
~~~
![image](https://github.com/user-attachments/assets/9e6b2e86-7545-426e-97e7-e0cbb4578783)
~~~
chip2,p, _, _=chi2_contingency(contingency_table)
print(f"Chi-square statistic: {chip2}")
print(f"P-value: {p}")
~~~
![image](https://github.com/user-attachments/assets/b9c2bc9f-4c78-4d04-b501-a9fb06695eba)
~~~
import pandas as pd 
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif 

data = { 
'Feature1': [1, 2, 3, 4, 5], 
'Feature2': ['A', 'B', 'C', 'A', 'B'], 
'Feature3': [0, 1, 1, 0, 1], 
'Target': [0, 1, 1, 0, 1] 
} 
df = pd.DataFrame(data) 

x= df[['Feature1', 'Feature3']] 
y = df['Target'] 
 
selector = SelectKBest(score_func=mutual_info_classif, k=1) 
X_new = selector.fit_transform(x, y)

selected_feature_indices = selector.get_support(indices=True) 


selected_features = X.columns[selected_feature_indices] 
print("Selected Features:") 
print(selected_features)
~~~
![image](https://github.com/user-attachments/assets/3147b8cf-146d-402b-b446-35eafd52d896)

# RESULT:
    Thus,The given data is read and performed Feature Scaling and Feature Selection process and saved the data to a file.


