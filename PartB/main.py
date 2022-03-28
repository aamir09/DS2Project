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
from sklearn.metrics import mean_squared_error



dirlist=os.listdir('A:/Datasets/DS2/')

masterData=cmd.createMasterDataSet(dirlist)

infoDf=cmd.getDataInfo(masterData)

#Create X and Y 

X=masterData.drop('Price',axis=1)
y=masterData['Price']

# masterData.to_csv('masterData.csv',index=None)

print(infoDf)

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=2022)

### PIPELINE INTEGRATION ###

dropcols=['Location']

remCols=masterData.drop(dropcols,axis=1).columns[1:]

outlierCols=['Area']

cleaningPipe=Pipeline([
    ('dropColumns',pc.dropColumns(dropcols)),
    ('nullValueReplacer',pc.replaceWithNan()),
    ('AddingFeatures',pc.addBin()),
    ('imputerNullValues',pc.categoricalImputer(remCols)),
    ('outlierHandling',pc.outlierHandling(outlierCols)),
    ('oneHotEncodeCity',pc.encoder(['city'])),
    ('addIqrFeature',pc.AddIQR())
])



masterPipeline=Pipeline([
    ('cleaning',cleaningPipe),
    ('model',pc.trainModels())
])

res=masterPipeline.fit_transform(X_train,y_train)

X_val=masterPipeline.transform(X_val)

X_train=res['X_train']
lreg=res['linearModel']
nn=res['neuralNetwork']['model']
tree=res['tree']

y_val=np.log(y_val)
lregMse=mean_squared_error(y_val,lreg.predict(X_val))
nnMse=mean_squared_error(y_val,nn.predict(np.asarray(X_val).astype('float32')))
treeMSe=mean_squared_error(y_val,tree.predict(X_val))

print('The mse of Linear Model is: ',lregMse)
print('The mse of Neural Network is: ', nnMse)
print('The mse of Random Forest Model is: ',treeMSe)


