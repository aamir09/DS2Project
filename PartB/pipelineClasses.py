# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import  StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

#Creating Pipeline Classes


#Dropping Columns
class dropColumns(BaseEstimator,TransformerMixin):
    def __init__(self,cols):
        #Calling the constructor method of both base classes 
        super().__init__()

        #Creating a variable to store columns to be dropped
        self.dropCols=cols

    def fit(self, X,y=None):
        #Pipelines fit  function
        nan=X.isna().sum()/X.shape[0]
        nan=nan[nan>0.55]
        drop=nan.index
        self.dropCols.extend(drop)
        print(f'Dropping Columns : {self.dropCols}')

        return self
    
    def transform(self,X):
        X=X.drop(columns=list(self.dropCols),axis=1)
        return X

#Imputation 

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self,cols):
        super().__init__()

        self.cols=cols
    
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        imputer=IterativeImputer(random_state=2022)
        XImputed=imputer.fit_transform(X[self.cols])
        XNew=pd.DataFrame(XImputed,columns=self.cols)
        return XNew



class standardize(BaseEstimator, TransformerMixin):

    def __init__(self,cols):
        super().__init__()
        self.cols=cols
    
    def fit(self,X,y=None):
        return self

    def transform(self,X):

        scaler=StandardScaler()
        XScaled=scaler.fit_transform(X[self.cols])
        return XScaled

class encoder(BaseEstimator, TransformerMixin):

    def __init__(self,cols):
        super().__init__()
        self.cols=cols
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        XNew=pd.DataFrame()
        for col in self.cols:
            dummies=pd.get_dummies(X[col])
            X=X.drop(col,axis=1)
            XNew=pd.concat([XNew,dummies],axis=1)
        return XNew.values


class categoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self,cols):
        super().__init__()

        self.cols=cols

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        imputer =SimpleImputer(strategy='most_frequent')
        XImputed=imputer.fit_transform(X[self.cols])
        return pd.DataFrame(XImputed,columns=self.cols)

#Nan Replacer
class replaceWithNan(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        return X.replace(9,np.nan)



        