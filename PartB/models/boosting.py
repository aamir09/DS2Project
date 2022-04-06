from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import catboost as cb

def gradientBoost(X,y):
    gboost=GradientBoostingRegressor(random_state=11)
    params={'max_depth':[1,2,3],'n_estimators':[100,300,500,1000]}
    search=GridSearchCV(estimator=gboost,param_grid=params,scoring='r2',cv=3,return_train_score=True)
    search.fit(X,y)
    print('Best parameters for Gradient Boosting are:', search.best_params_)
    return search.best_estimator_

def catboost(X,y):
    cboost=cb.CatBoostRegressor(verbose=False,silent=True,logging_level='Silent')
    params={'learning_rate':[0.1,0.5,0.01,0.05],'iterations':[100,300,500,1000],'depth':[1,2]}
    search=GridSearchCV(estimator=cboost,param_grid=params,scoring='r2',cv=3,return_train_score=True)
    search.fit(X,y)
    print('Best parameters for CatBoost are:', search.best_params_)
    return search.best_estimator_

