from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd 
import pickle
from sklearn.metrics import mean_squared_error,r2_score


def getBestEstimator(X,y):
    forest=BaggingRegressor(oob_score=True,n_jobs=-1,random_state=11,base_estimator=DecisionTreeRegressor(max_depth=9))
    params={'n_estimators':[50,100,125,1000]}
    search=GridSearchCV(estimator=forest,param_grid=params,scoring='r2',cv=3,return_train_score=True)
    search.fit(X,y)
    print('Best parameters for bagging model are:', search.best_params_)
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

# performanceMatrix={}
# trainMse=mean_squared_error(y_train,estimator.predict(X_train))
# testMse=mean_squared_error(y_val,estimator.predict(X_val))
# trainR2=r2_score(y_train,estimator.predict(X_train))
# testR2=r2_score(y_val,estimator.predict(X_val))
# performanceMatrix['bagging']={'MSE':{'train':trainMse,'test':testMse},'R2':{'train':trainR2,'test':testR2}}

# with open('PartB/models/BaggingperformanceMatrix.pickle','wb') as f:
#     pickle.dump(performanceMatrix,f,protocol=pickle.HIGHEST_PROTOCOL)




