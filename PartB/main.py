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

print(infoDf)

dropcols=['Price','Area','Location']

remCols=masterData.drop(dropcols,axis=1).columns

cleaningPipe=Pipeline([
    ('dropColumns',pc.dropColumns(dropcols)),
    ('nullValueReplacer',pc.replaceWithNan()),
    ('imputerNullValues',pc.Imputer(remCols)),
])
masterPipeline=Pipeline([
    ('cleaning',cleaningPipe),
    ('model',pc.trainModels())
])

# masterPipeline.fit(masterData,masterData['Price'])
# X_new=masterPipeline.score(masterData,masterData['Price'])
# print(X_new)

print(masterPipeline.fit_transform(masterData,masterData['Price']))