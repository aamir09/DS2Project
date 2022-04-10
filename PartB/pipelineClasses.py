# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import  StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from models import linearRegression as lr
from models import neuralNetwork as nn
from models import rforest as rf
from models import decisionTree as dt
from models import boosting as boost
from models import knn as knn
from models import bagging
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import os 

#Creating Pipeline Classes


#Dropping Columns
class dropColumns(BaseEstimator,TransformerMixin):
    '''Returns a data frame after removing the specifeid columns.
       parameters: 
       cols -> list: The columns to be removed
    '''
    def __init__(self,cols):
        #Calling the constructor method of both base classes 
        super().__init__()

        #Creating a variable to store columns to be dropped
        self.dropCols=cols

    def fit(self, X,y=None):
        #Pipelines fit  function
        return self
    
    def transform(self,X):
        print(f'Dropping Columns : {self.dropCols}')
        X=X.drop(columns=list(self.dropCols),axis=1)
        print('Is the index of our data frame unique ',X.index.is_unique)
        return X
    
    def fit_transform(self,X,y=None):
        print(f'Dropping Columns : {self.dropCols}')
        X=X.drop(columns=list(self.dropCols),axis=1)
        return X


class standardize(BaseEstimator, TransformerMixin):
    ''' Returns a standardized version of dataframe. The standardization takes place for the user given columns.
        parameters:
        cols-> list: Columns to be standardized in the dataframe.
    '''

    def __init__(self,cols):
        super().__init__()
        self.cols=cols
        self.scaler=StandardScaler()
    
    def fit(self,X,y=None):
        return self

    def fit_transform(self,X,y=None):
        XScaled=self.scaler.fit_transform(X[self.cols])
        for i,j in enumerate(self.cols):
            X.loc[:,j]=XScaled[:,i]
        return X

    def transform(self,X):
        XScaled=self.scaler.transform(X[self.cols])
        for i,j in enumerate(self.cols):
            X.loc[:,j]=XScaled[:,i]
        return X

class encoder(BaseEstimator, TransformerMixin):
    ''' Returns a dataframe inclusive if onehot encoded variables of the specified columns.
        parameters: 
        cols->list: Columns for which one hot encodeing is to be done. It drops the columns after creating it dummies.
    '''

    def __init__(self,cols):
        super().__init__()
        self.cols=cols
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        for col in self.cols:
            dummies=pd.get_dummies(X[col])
            X=X.drop(col,axis=1)
            X=pd.concat([X,dummies],axis=1)
        return X

    def fit_transform(self,X,y=None):
        for col in self.cols:
            dummies=pd.get_dummies(X[col])
            X=X.drop(col,axis=1)
            X=pd.concat([X,dummies],axis=1)
        return X



#Nan Replacer
class replaceWithNan(BaseEstimator, TransformerMixin):
    ''' Returns a dataframe with replacing missing indicator values by np.nan'''

    def __init__(self):
        super().__init__()

    def fit(self,X,y=None):
        return self

    def fit_transform(self,X,y=None):
        return X.replace(9,np.nan)
    
    def transform(self,X):
        return X.replace(9,np.nan)



class trainModels(BaseEstimator, TransformerMixin):
    ''' Returns a dictionary of trained models and train data on fit_transform and transforms the validation sets when transform function is called upon.
    '''
    def __init__(self):
        super().__init__()
        self.linearModel=None
        self.neuralNetwork=None
        self.forest=None
        self.tree=None
        self.gboost=None
        self.cboost=None
        self.knn=None
        self.xgboost=None
        self.bagging=None

    def fit(self,X,y=None):
        return self
    
    def fit_transform(self,X,y):
        #Train Linear Model
        y=np.log(y)
        print("Training Linear Rgression Model")
        self.linearModel=lr.trainModel(X,y)

        #Train Neural Network
        optimizer=tf.keras.optimizers.Adam(1e-3)
        loss='mse'
        metrics=['mse']
        print(X.shape)
        print('Training Neural Network Model')
        self.neuralNetwork=nn.trainNn(np.asarray(X).astype('float32'),y,optimizer=optimizer,loss=loss,metrics=metrics)
        print('Neural Network Training Complete!')
        print('Training Random Forest  Regressor Model')
        self.forest=rf.getBestEstimator(X,y)
        print('Training Decision Tree Regressor Model')
        self.tree=dt.decisionTree(X,y)
        print('Training K-Nearest Neighbor Regressor Model')
        self.knn=knn.knn(X,y)
        print('Training Bagging Model')
        self.bagging=bagging.getBestEstimator(X,y)
        print('Training Gradient Boosting Regressor Model')
        self.gboost=boost.gradientBoost(X,y)
        print('Training Cat Boost Regressor Model')
        self.cboost=boost.catboost(X,y)
        # print('Training XgBoost Model')
        # self.xgboost=boost.xgboost(X,y)
        print('All models training Complete!!')
        return {'X_train':X,'linearModel':self.linearModel,'neuralNetwork':self.neuralNetwork,'forest':self.forest,'tree':self.tree,'knn':self.knn,'bagging':self.bagging,'gboost':self.gboost,'cboost':self.cboost}

    def transform(self,X):
        return X


class outlierHandling(BaseEstimator, TransformerMixin):
    ''' Deals with ouliers in the given columns and return a dataframe with these mdoification. The outliers below the 25% qunatile are replaced with 
        it and the oulier above 75% quantile are replaced with the 75% quantile value. 
        parameters:
        cols->list: A list of column name that contains outliers.
    '''
    def __init__(self,cols):
        super().__init__()
        self.cols = cols
        self.limits={}
    
    def countOutliers(self,feature):
        lowerQuantile,higherQuantile = np.quantile(feature,[0.25,0.75])
        IQR=higherQuantile-lowerQuantile
        lowerLimit=lowerQuantile-1.5*IQR
        upperLimit=higherQuantile+1.5*IQR
        total=len(feature[(feature>=upperLimit) | (feature<=lowerLimit)])
        return total,lowerLimit, upperLimit

    def fit(self,X):
        return self

    def fit_transform(self,X,y=None):

        for i in self.cols:
            total,ll,ul=self.countOutliers(X[i])
            print(f'Number of outliers in feature {i} is: {total}')
            self.limits[i]={'ul':ul,'ll':ll}
            X.loc[X[i]<ll,i]=ll
            X.loc[X[i]>ul,i]=ul
        return X

    def transform(self,X):
        for i in self.cols:
            X.loc[X[i]<self.limits[i]['ll'],i]=self.limits[i]['ll']
            X.loc[X[i]>self.limits[i]['ul'],i]=self.limits[i]['ul']
        return X



class addBin(BaseEstimator, TransformerMixin):
    '''Returns a data frame with AreaType variable being added to the dataframe on the bases if the Area of the House.
    1  Small
    2  Medium
    3  Large
    4  Very Large '''
    def __init__(self):
        super().__init__()
        self.tf=None
        self.ff=None
        self.sf=None

    def fit(self,X):
        return X
    
    def fit_transform(self,X,y=None):
        print(X.index.is_unique)
        self.tf,self.ff,self.sf=np.quantile(X['Area'],[0.25,0.5,0.75])
        X['AreaType']=None
        X.loc[X['Area']<=self.tf,'AreaType']=1 #Small
        X.loc[(X['Area']>self.tf) & (X['Area']<=self.ff),'AreaType']= 2 #Medium
        X.loc[(X['Area']>self.ff) & (X['Area']<=self.sf),'AreaType']=3 #Large
        X.loc[X['Area']>self.sf,'AreaType']=4 #VeryLarge

        return X

    def transform(self,X):
        X['AreaType']=None
        X.loc[X['Area']<=self.tf,'AreaType']=1 #Small
        X.loc[(X['Area']>self.tf) & (X['Area']<=self.ff),'AreaType']= 2 #Medium
        X.loc[(X['Area']>self.ff) & (X['Area']<=self.sf),'AreaType']=3 #Large
        X.loc[X['Area']>self.sf,'AreaType']=4 #VeryLarge   

        return X


class customImputer(BaseEstimator, TransformerMixin):
    ''' Return a dataframe of imputed nan values. The strategy to impute the Nan values is for each feature calculaate the mode for each AreaType, check in which AreaType the sample lies
        and fill the Nan value with the mode of that feature corresponsing to that AreaType
        
        paramters:
        cols->list: A list of column conatining Null values
        '''

    def __init__(self,cols):
        super().__init__()
        self.statistics = {}
        self.cols=cols
        self.bed=None

    def calculateStatistic(self,X,feature):
        bin1=X[X['AreaType']==1].dropna()[feature].mode()
        bin2=X[X['AreaType']==2].dropna()[feature].mode()
        bin3=X[X['AreaType']==3].dropna()[feature].mode()
        bin4=X[X['AreaType']==4].dropna()[feature].mode()
        self.statistics[feature]={'bin1':bin1,'bin2':bin2,'bin3':bin3,'bin4':bin4}

    def fit(self,X):
        return self

    def fit_transform(self,X,y=None):
        for i in self.cols:
            index=[]
            if i.endswith('Bedrooms'):
                arr=np.zeros(shape=X.shape[0])
                idx=X.loc[(X[i].isna())].index
                toReplace=X[~X.index.isin(idx)].index
                m=X[i].mode()
                self.bed=m
                arr[idx]=self.bed
                arr[toReplace]=X.loc[toReplace,i]
                X.loc[:,i]=arr
                continue
            self.calculateStatistic(X,i)
            arr=np.zeros(shape=X.shape[0])
            for num,key in enumerate(self.statistics[i]):
                mode =self.statistics[i][key]
                idx=X.loc[(X[i].isna()) & (X['AreaType']==num+1)].index
                if len(idx)>0: 
                    arr[idx]=mode
                    index.extend(idx)
            if len(index)>0:
                toReplace=X[~X.index.isin(index)].index
                arr[toReplace]=X.loc[toReplace,i]
                X.loc[:,i]=arr
        print(X.isna().sum())
        return X

    def transform(self,X):
        for i in self.cols:
            index=[]
            if i.endswith('Bedrooms'):
                arr=np.zeros(shape=X.shape[0])
                idx=X.loc[(X[i].isna())].index
                toReplace=X[~X.index.isin(idx)].index
                arr[idx]=self.bed
                arr[toReplace]=X.loc[toReplace,i]
                X.loc[:,i]=arr
                continue
            arr=np.zeros(shape=X.shape[0])
            for num,key in enumerate(self.statistics[i]):
                mode =self.statistics[i][key]
                idx=X.loc[(X[i].isna()) & (X['AreaType']==num+1)].index
                if len(idx)>0: 
                    arr[idx]=mode
                    index.extend(idx)
            if len(index)>0:
                toReplace=X[~X.index.isin(index)].index
                arr[toReplace]=X.loc[toReplace,i]
                X.loc[:,i]=arr
        return X
        
class AddHQLI(BaseEstimator, TransformerMixin):
        '''Returns a dataframe with HQLI feature added to the dataframe. '''
        def __init__(self):
            super().__init__()
            self.qhi=None
            self.bai=None
            self.ai=None
            self.iqr=None

        def fit(self,X,y):
            return self

        def fit_transform(self,X,y):

            print(os.getcwd())
            QHI=pd.read_csv('PartB/QHI.csv',index_col='city')
            AI=pd.read_csv('PartB/AI.csv',index_col='city')
            BAI=pd.read_csv('PartB/BAI.csv',index_col='city')
            self.qhi=self.weighted_sum(QHI)
            self.bai=self.weighted_sum(BAI)
            self.ai=self.weighted_sum(AI)
            self.iqr={}
            for i in AI.index:
                self.iqr[i]=self.qhi[i]+self.bai[i]+self.ai[i]
            X['HQLI']=1
            X.loc[X['Delhi']==1,'HQLI']=self.iqr['DELHI']
            X.loc[X['Kolkata']==1,'HQLI']=self.iqr['KOLKATA']
            X.loc[X['Chennai']==1,'HQLI']=self.iqr['CHENNAI']
            X.loc[X['Hyderabad']==1,'HQLI']=self.iqr['HYDERABAD']
            X.loc[X['Mumbai']==1,'HQLI']=self.iqr['MUMBAI']
            X.loc[X['Bangalore']==1,'HQLI']=self.iqr['BANGLORE']

            return X

        def transform(self,X):
            X['HQLI']=None
            X.loc[X['Delhi']==1,'HQLI']=self.iqr['DELHI']
            X.loc[X['Kolkata']==1,'HQLI']=self.iqr['KOLKATA']
            X.loc[X['Chennai']==1,'HQLI']=self.iqr['CHENNAI']
            X.loc[X['Hyderabad']==1,'HQLI']=self.iqr['HYDERABAD']
            X.loc[X['Mumbai']==1,'HQLI']=self.iqr['MUMBAI']
            X.loc[X['Bangalore']==1,'HQLI']=self.iqr['BANGLORE']

            return X



        def weighted_sum(self,data):
            np.random.seed(2022)
            result={}
            weights=np.asarray(np.random.normal(0,1,data.shape[1]))
            for i in data.index:
                array=data.loc[i].values
                print(array)
                result[i]=np.sum(weights*array)
            return result




#### TEST #################################



# df=pd.read_csv('masterData.csv')
# df1=pd.read_csv('masterDataCleaned.csv')
# df['AreaType']=df1['AreaType']
# df=df.replace(9,np.nan)
# c=customImputer(df.columns)

# nanSample=np.random.randint(1,df.shape[0]-1,size=500)
# X=df.drop('Price',axis=1)
# X.iloc[nanSample]=np.nan

# print(c.fit_transform(df))



# df.loc[(df['Sofa'].isna()) & (df['AreaType']==4),'Sofa']=5

