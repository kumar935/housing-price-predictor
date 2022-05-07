import math
from flask_cors import CORS, cross_origin
from flask import Flask, jsonify, request
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_classif

print('############ LOGS START ############')

data = pd.read_csv('./Bengaluru_House_Data.csv')
print('############ printing data.info(): ')
data.info()

print("############ dropping area_type, availability, balcony, society as these won't have a significant enough effect on price")
data = data.drop(['area_type', 'availability', 'balcony', 'society'], axis=1)

data.isna().sum()  # detect missing values for cols
print('############ replacing missing values')
data = data.dropna()  # replace missing values


print('############ Adding new feature(integer) for bhk (Bedrooms Hall Kitchen) because size has string value')
# Add new feature(integer) for bhk (Bedrooms Hall Kitchen)
data['BHK'] = data['size'].apply(lambda x: int(x.split(' ')[0]))


def isfloat(x):
    try:
        float(x)
    except:
        return False
    return True
# we see from this there are some non float values for
data[~data['total_sqft'].apply(isfloat)].head(10)
def convert_sqft_tonum(x):
    token = x.split('-')
    if len(token) == 2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None
data = data.copy()
print('############ convert the non float values to float by averaging them for sq_ft because total_sqft has range string values')
# convert the non float values to float by averaging them
data['total_sqft'] = data['total_sqft'].apply(convert_sqft_tonum)


# Add new feature called price per square feet
data1 = data.copy()
print('############ Add new feature called price per square feet')
data1['price_per_sqft'] = data1['price']*1000000/data1['total_sqft']
data1.head()

# loc stuff: just printing the locs with their count in descending

data1.location = data1.location.apply(lambda x: x.strip())
location_stats = data1.groupby('location')['location'].agg(
    'count').sort_values(ascending=False)
# location_stats

# locations that have count less than 10
locationlessthan10 = location_stats[location_stats <= 10]
# locationlessthan10

print("############ Before removing locations that have count less than 10, because with so less count our prediction wouldn't be accurate")
print(len(data1.location.unique()))
# convert these less than 10 values to 'other'
data1.location = data1.location.apply(
    lambda x: 'other' if x in locationlessthan10 else x)
print('############ After removing locations that have count less than 10')
print(len(data1.location.unique()))

# sq ft per bhk < 300

data1[data1.total_sqft/data1.BHK < 300].head()

data2 = data1[~(data1.total_sqft/data1.BHK < 300)]

# outlier removal

data2["price_per_sqft"].describe().apply(lambda x: format(x, 'f'))


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft) # price per sq feet mean for each location
        st = np.std(subdf.price_per_sqft) # price per sq feet std dev for each location
        reduced_df = subdf[(subdf.price_per_sqft > (m-st))
                           & (subdf.price_per_sqft < (m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


data3 = remove_pps_outliers(data2)


data4 = data3.drop(['size', 'price_per_sqft'], axis='columns')
# data4

######## CHECK CORRELATION MATRIX ##########
print('############ Printing correlation matrix:')
print(data4.corr())
print('############ 1.96/(n^0.5)')
print(1.96/math.sqrt(len(data4.values)))
######## CHECK CORRELATION MATRIX ##########

dummies = pd.get_dummies(data4.location)
print('############ since area column does not have a numeric value we convert each of its value to columns with value 0 or 1')
data7 = pd.concat([data4, dummies.drop(
    'other', axis='columns')], axis='columns')

data8 = data7.drop('location', axis='columns')

X = data8.drop('price', axis='columns')

y = data8.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('############ Starting linear regression')
model = LinearRegression()
model.fit(X_train, y_train)

print('############ Printing R^2 score')
Rsq = model.score(X_test, y_test)
print(Rsq)

print('############ Anova analysis: F and pval')
F, pval = f_classif(X, y)

features_to_show = 3
print(pd.DataFrame([F[0:features_to_show], pval[0:features_to_show]], [
      'f', 'p'], X.columns[0:features_to_show]))



def price_predict(location, sqft, bath, BHK):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]


print('############# final columns: ')
print(len(X.columns))


# Using flask to make an api
# import necessary libraries and functions

# creating a Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@ app.route('/predict', methods=['GET'])
@ cross_origin()
def predict():
    args = request.args
    price = price_predict(
        args['location'], args['sqft'], args['bath'], args['BHK'])
    return str(price)


# driver function
if __name__ == '__main__':

    app.run(debug=True)
