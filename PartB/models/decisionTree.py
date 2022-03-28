from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def getBestEstimator(X,y):
    forest=RandomForestRegressor(oob_score=True,n_jobs=-1)
    params={'max_depth':[i for i in range(1,11)],'max_features':['log2']}
    search=GridSearchCV(estimator=forest,param_grid=params,scoring='r2',cv=5,return_train_score=True)
    search.fit(X,y)
    print('Best parameters for random forest are:', search.best_params_)
    return search.best_estimator_