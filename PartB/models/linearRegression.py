from sklearn.linear_model import LinearRegression

def trainModel(X,y):
    lr=LinearRegression()
    model=lr.fit(X,y)
    return model


