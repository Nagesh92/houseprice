import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('housing.csv',sep=',')
data

data = pd.DataFrame(data)
data

data.info()

data.describe()

data.dtypes

miss_val_perc = (data.isna().sum()/len(data))*100
miss_val_perc

num_cols = data.select_dtypes(include=['int64']).values
num_cols

cat_cols = data.select_dtypes(include=['object'])
cat_cols

data['price'] = np.log10(data['price'])
data['price']
plt.hist(data['price'])
plt.show()

data['area'] = data['price']/max(data['price'])
data['area']
plt.hist(data['area'])
plt.show()

cat_cols

le = LabelEncoder()
data.loc[:,['mainroad', 'guestroom', 'basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']]=\
data.loc[:,['mainroad', 'guestroom', 'basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']].apply(le.fit_transform)

data

x = data.drop(['price'],axis=1)
y = data.price

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state=3)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr= LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
r2_score_lr = r2_score(y_test,y_pred)
mse_lr = mean_squared_error(y_test,y_pred)
print(r2_score_lr)
print(mse_lr)

rf = RandomForestRegressor()
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)
r2_score_rf = r2_score(y_test,y_pred_rf)
mse_rf= mean_squared_error(y_test,y_pred_rf)
print(r2_score_rf)
print(mse_rf)

a= data.to_csv('revised.csv')
a


import pickle
pickle.dump(lr,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0.9906558979743387,4,1,2,1,1,1,0,1,2,0,0]]))




