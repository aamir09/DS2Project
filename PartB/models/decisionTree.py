from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def decisionTree(X,y):
    tree=DecisionTreeRegressor(random_state=11)
    params={'max_depth':[i for i in range(1,12)]}
    search=GridSearchCV(estimator=tree,param_grid=params,scoring='r2',cv=5,return_train_score=True)
    search.fit(X,y)
    print('Best parameters for Decision Tree are:', search.best_params_)
    return search.best_estimator_