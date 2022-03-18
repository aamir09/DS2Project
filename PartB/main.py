import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
import createMasterData as cmd
import pipelineClasses as pc
from sklearn.pipeline import Pipeline


dirlist=os.listdir('A:/Datasets/DS2/')

masterData=cmd.createMasterDataSet(dirlist)

infoDf=cmd.getDataInfo(masterData)

print(infoDf)

dropcols=['Price','Area','Location']

cleaningPipe=Pipeline([
    ('nullValueReplacer',pc.replaceWithNan()),
    ('imputerNullValues',pc.SimpleImputer(strategy='most_frequent')),
])
