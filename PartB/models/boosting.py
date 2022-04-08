from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import catboost as cb
import xgboost as xgb
import pandas as pd

def gradientBoost(X,y):
    gboost=GradientBoostingRegressor(random_state=11)
    params={'max_depth':[1,2,3],'n_estimators':[100,300,500,1000]}
    search=GridSearchCV(estimator=gboost,param_grid=params,scoring='r2',cv=3,return_train_score=True,verbose=2)
    search.fit(X,y)
    print('Best parameters for Gradient Boosting are:', search.best_params_)
    return search.best_estimator_

def catboost(X,y):
    cboost=cb.CatBoostRegressor(silent=True)
    params={'learning_rate':[0.1,0.5,0.01,0.05],'iterations':[100,300,500,1000],'depth':[1,2]}
    search=GridSearchCV(estimator=cboost,param_grid=params,scoring='r2',cv=3,return_train_score=True,verbose=2)
    search.fit(X,y)
    print('Best parameters for CatBoost are:', search.best_params_)
    return search.best_estimator_


def xgboost(X,y):
    xgboost=xgb.XGBRegressor(objective ='reg:linear',enable_cateorical=True)
    params={'learning_rate':[0.1,0.5,0.01,1],'n_estimators':[10,30,100,500],'colsample_bytree':[0.3,0.5],
                'max_depth':[1,2,3,4,5], 'alpha': [0.1,1,0.01,10]}
    search=GridSearchCV(estimator=xgboost,param_grid=params,scoring='r2',cv=3,return_train_score=True,verbose=1)
    search.fit(X,y)
    print('Best parameters for XgBoost are:', search.best_params_)
    return search.best_estimator_

