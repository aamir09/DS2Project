import pandas as pd
import numpy as np
import os
import createMasterData as cmd


dirlist=os.listdir('A:/Datasets/DS2/')

masterData=cmd.createMasterDataSet(dirlist)

infoDf=cmd.getDataInfo(masterData)

print(infoDf)
