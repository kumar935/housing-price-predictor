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

# data cleaning
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
    # return df # commenting outlier stuff
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st))
                           & (subdf.price_per_sqft < (m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


data3 = remove_pps_outliers(data2)
# data3.shape

# Outlier Removal


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.BHK == 2)]
    bhk3 = df[(df.location == location) & (df.BHK == 3)]
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='Blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, color='green',
                marker='+', label='3 BHK', s=50)
    plt.xlabel('Total Square Foot')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()


plot_scatter_chart(data3, "Rajaji Nagar")


def remove_bhk_outliers(df):
    return df  # commenting outlier stuff
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_sats = {}
        for BHK, BHK_df in location_df.groupby('BHK'):
            bhk_sats[BHK] = {
                'mean': np.mean(BHK_df.price_per_sqft),
                'std': np.std(BHK_df.price_per_sqft),
                'count': BHK_df.shape[0]
            }
        for BHK, BHK_df in location_df.groupby('BHK'):
            stats = bhk_sats.get(BHK-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices, BHK_df[BHK_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')


data4 = remove_bhk_outliers(data3)


more_outlier_removal = 0
if more_outlier_removal:
    plot_scatter_chart(data4, "Rajaji Nagar")
    plt.rcParams['figure.figsize'] = (20, 15)
    plt.hist(data4.price_per_sqft, rwidth=0.6)
    plt.xlabel("Price Per Square Foor")
    plt.ylabel("Count")
    # outlier removal using bathroom features
    plt.rcParams['figure.figsize'] = (20, 15)
    plt.hist(data4.bath, rwidth=0.6)
    plt.xlabel("Number Of Bathroom")
    plt.ylabel("Count")


# commenting outlier stuff - data5 = data4[data4.bath < data4.BHK+2]
data5 = data4
# data5.shape


data6 = data5.drop(['size', 'price_per_sqft'], axis='columns')
# data6

######## CHECK CORRELATION MATRIX ##########
print('############ Printing correlation matrix:')
print(data6.corr())
print('############ 1.96/(n^0.5)')
print(1.96/math.sqrt(len(data6.values)))
######## CHECK CORRELATION MATRIX ##########

dummies = pd.get_dummies(data6.location)
print('############ since area column does not have a numeric value we convert each of its value to columns with value 0 or 1')
data7 = pd.concat([data6, dummies.drop(
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

N = len(X.values)
M = len(X.columns)
fstat = (Rsq/(1-Rsq))*((N-M)/(M-1))
print('############ Printing F value')
print(fstat)

print('############ Anova analysis: F and pval')
F, pval = f_classif(X, y)
# print(F, pval)
# print('############')
features_to_show = 3
print(pd.DataFrame([F[0:features_to_show], pval[0:features_to_show]], [
      'f', 'p'], X.columns[0:features_to_show]))
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)
def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'],
                          cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
# find_best_model_using_gridsearchcv(X, y)


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
print(X.columns)


# Using flask to make an api
# import necessary libraries and functions

# creating a Flask app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.


@ app.route('/', methods=['GET', 'POST'])
def home():
    if(request.method == 'GET'):

        data = "hello world"
        return jsonify({'data': data})


# A simple function to calculate the square of a number
# the number to be squared is sent in the URL when we use GET
# on the terminal type: curl http://127.0.0.1:5000 / home / 10
# this returns 100 (square of 10)
@ app.route('/home/<int:num>', methods=['GET'])
def disp(num):

    return jsonify({'data': num**2})


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
