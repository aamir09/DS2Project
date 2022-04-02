import pandas as pd
import numpy as np
import os

# Creating a Single CSV File (Master Data)

def getPath(city:str)->str:
    '''It returns the path for the file in the local machine
    PARAMETERS
    city: name of the city 
    '''
    return f'A:/Datasets/DS2/{city}'


def createMasterDataSet(fileNames:list)->pd.DataFrame:
    ''' Concatenates all the csv files in a directory
        fileNames: Name of the files in the directory '''
    print('Importing the first dataset in the directory')
    master=pd.read_csv(getPath(fileNames[0]))
    master['city']=fileNames[0].split('/')[-1].replace('.csv','')
    print('Concatenating all the other dataframes to create master data')
    for i in fileNames[1:]:
        tempDf=pd.read_csv(getPath(i))
        tempDf['city']=i.split('/')[-1].replace('.csv','')
        master=pd.concat([master,tempDf],axis=0,ignore_index=True)
    print('Master Data Successfully Generated')
    return master


#Explore the master data 

def getDataInfo(data:pd.DataFrame)->pd.DataFrame:
    ''' Returns a dataframe containing information about columns of the data set for
        data: The dataset to get infomration of.'''
    info=pd.DataFrame()
    unique=[]
    nunique=[]
    nan=[]
    dtype=[]
    for i in data:
        unique.append(data[i].unique())
        nunique.append(data[i].nunique())
        nan.append(np.mean(data[i]==9))
        dtype.append(data[i].dtype)

    info['columns']=data.columns
    info['dataType']=dtype
    info['percntageNullValues']= np.round(nan,4)*100
    info['numberOfUniqueValues']= nunique
    info['uniqueValues']=unique
    

    return info


# Cleaning 
#0. Drop object columns
# 1. Replace 9 with nan 
#2. Impute nan Values 
#3. Nominal Encode area


#Modeling 

#Create 3 Models
# Linear R, Neural Net and Tree based 









