from doctest import master
import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
import createMasterData as cmd
import pipelineClasses as pc
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


dirlist=os.listdir('A:/Datasets/DS2/')

masterData=cmd.createMasterDataSet(dirlist)

infoDf=cmd.getDataInfo(masterData)

# masterData.to_csv('masterData.csv',index=None)

# print(infoDf)

dropcols=['Price','Location']

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
    ('cleaning',cleaningPipe)
    # ('model',pc.trainModels())
])
################## Do not De Comment ##########################
X=masterPipeline.fit_transform(masterData,masterData['Price'])
# X_new=masterPipeline.score(masterData,masterData['Price'])
# print(X_new)
##############################################################
# print(masterPipeline.fit_transform(masterData,masterData['Price']))

# X=cleaningPipe.fit_transform(masterData)
# print(X.isna().sum())
# X.to_csv('masterDataCleaned.csv')

print(X)