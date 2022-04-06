from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

def knn(X,y):
    knn=KNeighborsRegressor(n_jobs=-1)
    search=GridSearchCV(estimator=knn,param_grid={'n_neighbors':[i for i in range(1,11)]},scoring='r2',cv=3,return_train_score=True)
    search.fit(X, y)
    print('Best parameters for KNN Regressor are:', search.best_params_)
    return search.best_estimator_