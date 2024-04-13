#@author: SWAYAM SONI

#%%
import pandas as pd
import numpy as np
import os

df = pd.read_csv("_All_Cities_Cleaned.csv")
df.drop_duplicates(inplace = True)
df["price"]=np.log(df["price"])
df["area"]=np.log(df["area"])


df["seller_type"]=df["seller_type"].map({'OWNER':0,'AGENT':1,'BUILDER':2})
df["layout_type"]=df["layout_type"].map({'BHK':0,'RK':1})
df["property_type"]=df["property_type"].map({'Apartment':0,'Studio Apartment':1,'Independent House':2,'Independent Floor':3,'Villa':4,'Penthouse':5})
df["furnish_type"]=df["furnish_type"].map({'Furnished':0,'Semi-Furnished':1,'Unfurnished':2})
df["city"]=df["city"].map({'Ahmedabad':0,'Bangalore':1,'Chennai':2,'Delhi':3,'Hyderabad':4,'Kolkata':5,'Mumbai':6,'Pune':7})

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
el = label_encoder.fit(df["locality"])
encoded_localities = label_encoder.fit_transform(df["locality"])
print(encoded_localities)
d = {'localities':encoded_localities}
dum2=pd.DataFrame(data=d)

df2=pd.concat([df,dum2],axis=1)

df2.drop("locality",axis=1,inplace=True)

df2.dropna(inplace=True)
X = df2[["seller_type", "bedroom", "layout_type", "property_type", "area", "furnish_type", "bathroom", "city", "localities"]]
y=df2["price"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#%%
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('MS',model.score(X_test,y_test))
print(r2_score(y_test,y_pred))
print("6")


#%%         
from sklearn.svm import SVR
best_model = SVR(kernel="rbf",C=50000,epsilon=0.5,)
print("6")
best_model.fit(X_train, y_train)
print("6.5")
y_pred = best_model.predict(X_test)
print("7")
print('MS',best_model.score(X_test,y_test))
from sklearn.metrics import r2_score 
print(r2_score(y_test,y_pred))
#%%
from sklearn.neighbors import KNeighborsRegressor

# Create the KNeighborsRegressor model
neigh = KNeighborsRegressor(n_neighbors=5,weights='distance')

# Fit the model to the training data
neigh.fit(X_train, y_train)

# Make predictions on the test data
y_pred = neigh.predict(X_test)
print("7")
"""
from xgboost import XGBRegressor
from sklearn.metrics import r2_score 
model = XGBRegressor(n_estimators=300, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(r2_score(y_test,y_pred))
#%%
from joblib import dump
import pickle
pickle.dump(model,open("model.pkl","wb"))