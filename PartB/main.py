from doctest import master
import nntplib
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.compose import ColumnTransformer
import createMasterData as cmd
import pipelineClasses as pc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


################################ CREATING MASTER DATA ################################
dirlist=os.listdir('PartB/Datasets/')

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

################################ PIPELINE INTEGRATION ################################

dropcols=['Location']

remCols=list(masterData.drop(dropcols,axis=1).columns[1:])
remCols.remove('city')
remCols.remove('Area')

outlierCols=['Area']
contCols=['Area','HQLI']

cleaningPipe=Pipeline([
    ('dropColumns',pc.dropColumns(dropcols)),
    ('nullValueReplacer',pc.replaceWithNan()),
    ('AddingFeatures',pc.addBin()),
    ('imputerNullValues',pc.customImputer(remCols)),
    ('outlierHandling',pc.outlierHandling(outlierCols)),
    ('oneHotEncodeCity',pc.encoder(['city'])),
    ('addIqrFeature',pc.AddHQLI()),
    ('Standardization',pc.standardize(contCols))
],verbose=True)



masterPipeline=Pipeline([
    ('cleaning',cleaningPipe),
    ('model',pc.trainModels())
],verbose=True)


res=masterPipeline.fit_transform(X_train,y_train)
print(res)
X_val=masterPipeline.transform(X_val)
print(X_val)


################################ SAVING MODELS ################################

X_train=res['X_train']
lreg=res['linearModel']
nn=res['neuralNetwork']['model']
forest=res['forest']
history=res['neuralNetwork']['history']
tree=res['tree']
knn=res['knn']
gboost=res['gboost']
cboost=res['cboost']
bagging=res['bagging']


modelList={'lreg':lreg,'history':history,'forest':forest,'tree':tree,'bagging':bagging,'knn':knn,'gboost':gboost,'cboost':cboost}

for i in modelList:
    with open(f'PartB/models/savedModels/{i}.pickle','wb') as f:
        pickle.dump(modelList[i],f,protocol=pickle.HIGHEST_PROTOCOL)
        
modelJson=nn.to_json()
with open('PartB/models/savedModels/nueralNetwork.json','w') as file:
    file.write(modelJson)
nn.save_weights('PartB/models/savedModels/nueralNetworkWeights.h5')

################################ Calculating PERFORMANCE MATRIX ################################
y_val=np.log(y_val)
lregMse=mean_squared_error(y_val,lreg.predict(X_val))
nnMse=mean_squared_error(y_val,nn.predict(np.asarray(X_val).astype('float32')))
forestMse=mean_squared_error(y_val,forest.predict(X_val))
treeMse=mean_squared_error(y_val,tree.predict(X_val))
knnMse=mean_squared_error(y_val,knn.predict(X_val))
baggingMse=mean_squared_error(y_val,bagging.predict(X_val))
gboostMse=mean_squared_error(y_val,gboost.predict(X_val))
cboostMse=mean_squared_error(y_val,cboost.predict(X_val))



lregR2=r2_score(y_val,lreg.predict(X_val))
nnR2=r2_score(y_val,nn.predict(np.asarray(X_val).astype('float32')))
forestR2=r2_score(y_val,forest.predict(X_val))
treeR2=r2_score(y_val,tree.predict(X_val))
knnR2=r2_score(y_val,knn.predict(X_val))
baggingR2=r2_score(y_val,bagging.predict(X_val))
gboostR2=r2_score(y_val,gboost.predict(X_val))
cboostR2=r2_score(y_val,cboost.predict(X_val))


print('----------------------Performance on Test Data-----------------------------')
print()
print()
print('The mse of Linear Model is: ',lregMse)
print('The mse of Neural Network is: ', nnMse)
print('The mse of Random Forest Model is: ',forestMse)
print('The mse of Decision Tree Model is: ',treeMse)
print('The mse of Bagging Model is: ',baggingMse)
print('The mse of Gradient Boosting Model is: ',gboostMse)
print('The mse of Cat Boosting Model is: ',cboostMse)
print('The mse of KNN Regressor Model is: ',knnMse)
print()
print('The r2 Score of Linear Model is: ',lregR2)
print('The r2 Score of Neural Network is: ', nnR2)
print('The r2 Score of Random Forest Model is: ',forestR2)
print('The r2 Score of Decision Tree Model is: ',treeR2)
print('The r2 Score of Bagging Model is: ',baggingR2)
print('The r2 Score of Gradient Boosting Model is: ',gboostR2)
print('The r2 Score of Cat Boosting Model is: ',cboostR2)
print('The r2 Score of KNN Regressor Model is: ',knnR2)

################################ SAVING PERFORMANCE MATRIX ##############################
del modelList['history']

modelList['nn']=nn

performanceMatrix={}

for i in modelList:
    Xt=X_train
    Xv=X_val
    yt=np.log(y_train)
    if i=='nn':
        Xt=np.asarray(X_train).astype('float32')
        Xv=np.asarray(X_val).astype('float32')
    trainMse=mean_squared_error(yt,modelList[i].predict(Xt))
    testMse=mean_squared_error(y_val,modelList[i].predict(Xv))
    trainR2=r2_score(yt,modelList[i].predict(Xt))
    testR2=r2_score(y_val,modelList[i].predict(Xv))
    performanceMatrix[i]={'MSE':{'train':trainMse,'test':testMse},'R2':{'train':trainR2,'test':testR2}}

with open('PartB/models/performanceMatrix.pickle','wb') as f:
    pickle.dump(performanceMatrix,f,protocol=pickle.HIGHEST_PROTOCOL)



##### Creating Train-Test Files #####

train=pd.concat([X_train,np.log(y_train)],axis=1)
test=pd.concat([X_val,y_val],axis=1)

train.to_csv('S:/DS2Project/train.csv')
test.to_csv('S:/DS2Project/test.csv')