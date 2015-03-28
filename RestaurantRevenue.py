__author__ = 'Harsh'

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from datetime import datetime
#Load training data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# Find maximum data and subtract days from it to check how old hotel is

vec = DictVectorizer()

#Encode label
#labelencoder = preprocessing.LabelEncoder()
#train['City'] = labelencoder.fit_transform(train['City'])
#train['City Group'] = labelencoder.fit_transform(train['City Group'])
#train['Type'] = labelencoder.fit_transform(train['Type'])
#train['Open Date'] = labelencoder.fit_transform(train['Open Date'])

def diff_dates_2015(date_x):
  date_format = "%m/%d/%Y"
  x = datetime.strptime(date_x, date_format)
  y = datetime.strptime('01/01/2015', date_format)
  delta = y - x
  return delta.days

train['Open Date'] = train['Open Date'].apply(lambda x: diff_dates_2015(x))
test['Open Date'] = test['Open Date'].apply(lambda x: diff_dates_2015(x))

#Extract Features.to
train_new = vec.fit_transform(train[['City','City Group','Type']].T.to_dict().values()).todense()
test_new = vec.transform(test[['City','City Group','Type']].T.to_dict().values()).todense()

print train_new
print test_new

target = train['revenue']
train = train.drop('revenue',axis=1)

#test['City'] = labelencoder.fit_transform(test['City'])
#test['City Group'] = labelencoder.fit_transform(test['City Group'])
#test['Type'] = labelencoder.fit_transform(test['Type'])
#test['Open Date'] = labelencoder.fit_transform(test['Open Date'])

p = ['P' + str(i) for i in range(1,38)]
train_p = train[p]
test_p = test[p]

train = np.hstack((train_new,train_p))
test = np.hstack((test_new,test_p))

#Setup Random Forest
clf = RandomForestRegressor(n_estimators=200)
clf.fit(train,target)
print clf.feature_importances_
test_revenue = clf.predict(test)

sub = pd.read_csv('sampleSubmission.csv')
sub['Prediction'] = test_revenue
sub.to_csv('RandomForest.csv', index = False)
