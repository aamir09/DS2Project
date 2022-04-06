from sklearn.cluster import estimate_bandwidth
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd 
from sklearn.metrics import mean_squared_error,r2_score


def getBestEstimator(X,y):
    forest=RandomForestRegressor(oob_score=True,n_jobs=-1,random_state=11)
    params={'max_depth':[i for i in range(5,12)],'n_estimators':[50,100,125,150]}
    search=GridSearchCV(estimator=forest,param_grid=params,scoring='r2',cv=5,return_train_score=True)
    search.fit(X,y)
    print('Best parameters for random forest are:', search.best_params_)
    return search.best_estimator_

# train=pd.read_csv('train.csv')
# test=pd.read_csv('test.csv')

# X_train=train.drop('Price',axis=1)
# y_train=train['Price']

# X_val=test.drop('Price',axis=1)
# y_val=test['Price']

# estimator=getBestEstimator(X_train,y_train)

# print(r2_score(y_train,estimator.predict(X_train)))
# print(r2_score(y_val,estimator.predict(X_val)))
