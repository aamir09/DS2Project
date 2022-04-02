from doctest import master
import nntplib
import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
import createMasterData as cmd
import pipelineClasses as pc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score



dirlist=os.listdir('A:/Datasets/DS2/')

masterData=cmd.createMasterDataSet(dirlist)

infoDf=cmd.getDataInfo(masterData)

#Create X and Y 

X=masterData.drop('Price',axis=1)
y=masterData['Price']

# masterData.to_csv('masterData.csv',index=None)

print(infoDf)

X_train,X_val,y_train,y_val=train_test_split(X.values,y.values,test_size=0.2,random_state=2022)

X_train=pd.DataFrame(X_train,columns=X.columns)
X_val=pd.DataFrame(X_val,columns=X.columns)
y_train=pd.Series(y_train,name='Price')
y_val=pd.Series(y_val,name='Price')

### PIPELINE INTEGRATION ###

dropcols=['Location']

remCols=list(masterData.drop(dropcols,axis=1).columns[1:])
remCols.remove('city')
remCols.remove('Area')

outlierCols=['Area']

cleaningPipe=Pipeline([
    ('dropColumns',pc.dropColumns(dropcols)),
    ('nullValueReplacer',pc.replaceWithNan()),
    ('AddingFeatures',pc.addBin()),
    ('imputerNullValues',pc.customImputer(remCols)),
    ('outlierHandling',pc.outlierHandling(outlierCols)),
    ('oneHotEncodeCity',pc.encoder(['city'])),
    ('addIqrFeature',pc.AddHQLI())
],verbose=True)



masterPipeline=Pipeline([
    ('cleaning',cleaningPipe),
    # ('model',pc.trainModels())
],verbose=True)


# res=masterPipeline.fit_transform(X_train,y_train)
# print(res)
X_train=masterPipeline.fit_transform(X_train,y_train)
X_val=masterPipeline.transform(X_val)
print(X_val)


# X_train=res['X_train']
# lreg=res['linearModel']
# nn=res['neuralNetwork']['model']
# tree=res['tree']

y_val=np.log(y_val)
# lregMse=mean_squared_error(y_val,lreg.predict(X_val))
# nnMse=mean_squared_error(y_val,nn.predict(np.asarray(X_val).astype('float32')))
# treeMSe=mean_squared_error(y_val,tree.predict(X_val))

# lregR2=r2_score(y_val,lreg.predict(X_val))
# nnR2=r2_score(y_val,nn.predict(np.asarray(X_val).astype('float32')))
# treeR2=r2_score(y_val,tree.predict(X_val))

# print('The mse of Linear Model is: ',lregMse)
# print('The mse of Neural Network is: ', nnMse)
# print('The mse of Random Forest Model is: ',treeMSe)

# print('The r2 Score of Linear Model is: ',lregR2)
# print('The r2 Score of Neural Network is: ', nnR2)
# print('The r2 Score of Random Forest Model is: ',treeR2)




##### Creating Train-Test Files #####

train=pd.concat([X_train,np.log(y_train)],axis=1)
test=pd.concat([X_val,y_val],axis=1)

train.to_csv('S:/DS2Project/train.csv')
test.to_csv('S:/DS2Project/test.csv')