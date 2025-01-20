import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

#first we need to read our data.
X_all = pd.read_csv('./train.csv', index_col='Id')
X_test_all = pd.read_csv('./test.csv', index_col='Id')

#then we need to clear out data for accuracy for our model and clear outcomes.
#the subset cleared because of our target to House Price model. so this is important to not having the wrong feature.
X_all.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_all.SalePrice
X_all.drop(['SalePrice'], axis = 1, inplace = True)

#Excluding the 'object' because i want to use numerical.
X = X_all.select_dtypes(exclude=['object'])
X_test = X_test_all.select_dtypes(exclude=['object'])

X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8, test_size=0.2, random_state = 0)

print(X_train.describe)
print(X_train.shape)

#well we got missing values so we need to clear them out.

missing_data_cols = (X_train.isnull().sum())
print(missing_data_cols[missing_data_cols > 0])

#LotFrontage    212
#MasVnrArea       6
#GarageYrBlt     58
#so we have 3 missing column. I'll use RandomForest and MAE for handling missing values in advance of predicting.

#n_estimators refers to the total number of decision trees the model will create. Increasing the number of trees can improve the model's performance, but it also requires more computational resources (time and memory).
def model_score(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators= 100, random_state= 0)
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, pred)
    return mean_absolute_error(y_valid,pred)

missing_cols = [col for col in X_train.columns if X_train[col].isnull().any()]

dropped_X_train = X_train.drop(missing_cols, axis = 1)
dropped_X_valid = X_valid.drop(missing_cols, axis = 1)

#that's why created an def function for evaluate and see the outcome when RandomForestRegressor implemented out model, so the MAE score dropped a bit.
print(model_score(dropped_X_train, dropped_X_valid, y_train, y_valid))

#I'm going to use imputer for fill the missing values of median or avg value on dataset so we can evaluate our score much better.
impute = SimpleImputer(strategy='median')
impute_X_train = pd.DataFrame(impute.fit_transform(X_train))
impute_X_valid = pd.DataFrame(impute.transform(X_valid))

impute_X_train.columns = X_train.columns
impute_X_valid.columns = X_valid.columns

model = RandomForestRegressor(n_estimators=100, random_state= 0)
model.fit(impute_X_train, y_train)
pred_valid = model.predict(impute_X_valid)
print("MAE (Impute):")
print(mean_absolute_error(y_valid, pred_valid))

last_pred = SimpleImputer(strategy='median')
last_X_test = pd.DataFrame(last_pred.fit_transform(X_test), columns= X_test.columns)
pred_test_results = model.predict(last_X_test)
print(pred_test_results)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': pred_test_results})
output.to_csv('submission.csv', index=False)
