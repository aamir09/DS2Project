import streamlit as st
import numpy as np
import plotly as pt
import pandas as pd
st.set_page_config(
     page_title="Indian Housing Data",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )

st.markdown('<style> body{background-color:#ffff}</style>',unsafe_allow_html=True) 
title= '<h1 style="font-family:Courier; color:#FA8072; align:left;font-size: 3rem;">Exploration and Modelling of Indain Housing Data</h1>'
st.markdown(title,unsafe_allow_html=True) 

contrib_title= '<h2 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.6rem;">Contributors</h2>'
st.markdown(contrib_title,unsafe_allow_html=True) 
contributors='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Aamir Ahmad Ansari <a href="https://www.github.com" style="color=#fffff;text-decoration:none">GitHub </a>| <a href="https://www.github.com" style="color=#fffff;text-decoration:none">LinkedIn</a></li><ul>'
st.markdown(contributors,unsafe_allow_html=True) 
contributors='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Ibtisam <a href="https://www.github.com" style="color=#fffff;text-decoration:none">GitHub </a>| <a href="https://www.github.com" style="color=#fffff;text-decoration:none">LinkedIn</a></li><ul>'
st.markdown(contributors,unsafe_allow_html=True)
contributors='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Rohit Raj <a href="https://www.github.com" style="color=#fffff;text-decoration:none">GitHub </a>| <a href="https://www.github.com" style="color=#fffff;text-decoration:none">LinkedIn</a></li><ul>'
st.markdown(contributors,unsafe_allow_html=True)  
contributors='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Hemani Shah <a href="https://www.github.com" style="color=#fffff;text-decoration:none">GitHub </a>| <a href="https://www.github.com" style="color=#fffff;text-decoration:none">LinkedIn</a></li><ul>'
st.markdown(contributors,unsafe_allow_html=True) 

### Summary ###
summary_title= '<h2 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.6rem;">Summary</h2>'
st.markdown(summary_title,unsafe_allow_html=True) 
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ante in nibh mauris cursus mattis molestie. Vestibulum mattis ullamcorper velit sed ullamcorper morbi tincidunt. Id porta nibh venenatis cras sed felis. Ut ornare lectus sit amet est placerat in. Diam sollicitudin tempor id eu nisl nunc. Faucibus turpis in eu mi bibendum neque. Integer eget aliquet nibh praesent. Tellus molestie nunc non blandit massa enim nec dui. Rutrum quisque non tellus orci ac. Varius sit amet mattis vulputate enim. Ac auctor augue mauris augue. Nam aliquam sem et tortor consequat id. Mattis rhoncus urna neque viverra justo nec ultrices dui sapien. Viverra adipiscing at in tellus integer. Eget aliquet nibh praesent tristique magna sit amet purus. Faucibus purus in massa tempor nec feugiat.</p>'
st.markdown(summary,unsafe_allow_html=True) 

### EDA ###
eda_title= '<h2 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.6rem;">Exploratory Data Analysis</h2>'
st.markdown(eda_title,unsafe_allow_html=True) 
col1,col2=st.columns([0.5,0.5])
col1.image('app/images/goodSates.jpeg',caption='States with Good Housing Conditions',width=1200,use_column_width=True)
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ante in nibh mauris cursus mattis molestie. Vestibulum mattis ullamcorper velit sed ullamcorper morbi tincidunt. Id porta nibh venenatis cras sed felis. Ut ornare lectus sit amet est placerat in. Diam sollicitudin tempor id eu nisl nunc. Faucibus turpis in eu mi bibendum neque. Integer eget aliquet nibh praesent. Tellus molestie nunc non blandit massa enim nec dui. Rutrum quisque non tellus orci ac. Varius sit amet mattis vulputate enim. Ac auctor augue mauris augue. Nam aliquam sem et tortor consequat id. Mattis rhoncus urna neque viverra justo nec ultrices dui sapien. Viverra adipiscing at in tellus integer. Eget aliquet nibh praesent tristique magna sit amet purus. Faucibus purus in massa tempor nec feugiat.</p>'
col2.markdown(summary,unsafe_allow_html=True) 



### Modelling ###

model_title= '<h2 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.6rem;">Modelling</h2>'
st.markdown(model_title,unsafe_allow_html=True) 
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The objective of this section is to construct and compare at least 3 models to predict the pricing of houses from Metropolitan Indian Cities and perform feature engineering to increase the accuracy of your models. Some features to add would be House Quality Living Index by using the Census dataset or other features using the comprehensive Census dataset provided. The dataset we will be using comprises of the collection of prices of new and resale houses located in the metropolitan areas of India and the amenities provided for each house. With 40 explanatory variables describing various aspects of new and resale houses in the metropolitan areas of India, one can predict the final price of houses in these regions.The dataset can be found here,<a href="https://www.kaggle.com/ruchi798/housing-prices-in-metropolitan-areas-of-india/download" style="color=#fffff;text-decoration:none">link to the dataset.</a> </p>'
st.markdown(summary,unsafe_allow_html=True) 

# Dataset
dataset= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.4rem;">Dataset</h3>'
st.markdown(dataset,unsafe_allow_html=True) 
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The data set is an open source data set available at kaggle. The data contains information about the house and its amenities as mentioned above. The data is distributed in 5 files and each represent a metro city. Delhi, Mumbai, Bangalore, Chennai, Hyderabad and Kolkata are the ones that are present in the data. Price is our target variable and the rest 39 variables are accounted as feature variables. Lets take a peak at the dataset.</p>'
st.markdown(summary,unsafe_allow_html=True) 
table=pd.read_csv('masterData.csv',nrows=5)
st.dataframe(table)
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The above dataset is created after merging the 5 files, observe a variable city is also added to the data frame as an indicator that specifies the city from which the sample belong.</p>'
st.markdown(summary,unsafe_allow_html=True)



#Problems
problems= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.4rem;">Problems with the Dataset</h3>'
st.markdown(problems,unsafe_allow_html=True) 
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The following are the challenges faced with this dataset:</p>'
st.markdown(summary,unsafe_allow_html=True)

problem_1='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Area is the only continious variable, number of bedrooms is ordinal and the rest are binary variables.</li><ul>'
st.markdown(problem_1,unsafe_allow_html=True) 

problem_2='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">The binary variables have almost 70% null values each. The Nan values have been encoded as 9 for the binary variables.</li><ul>'
st.markdown(problem_2,unsafe_allow_html=True) 

problem_3='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">The feature Area is left skewed and contains several outliers, that has to be dealt with, shown in Figure 1.</li><ul>'
st.markdown(problem_3,unsafe_allow_html=True) 

col1, col2, col3 = st.columns([1,6,1])

with col1:
  st.write("")

with col2:
  st.image('app/images/AreaDistribution.jpeg',caption='Figure 1: Distribution of Feature Area')

with col3:
  st.write("")

### Proposed Solution ###
solution= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.4rem;">Proposed Solution</h3>'
st.markdown(solution,unsafe_allow_html=True) 

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The solution we are providing is one of an end-to-end pipeline which takes care of cleaning the data appropriately, adding features from the data as well as by augmentation and training the machine learning and deep learning models. To implement all of this we propose our own custom classes at each step of the pipeline. Figure 2 shows the outline of our proposed pipeline. Our solution is divided into 3 levels, Leve1: Creating Master Data, Level2: Cleaning Pipeline, Level3: Modelling. The end to end pipeline contains only level2 and level3, level1 is computed seperately. The workflow of the pipeline is different from conventional pipelines as we use the fit_transform method to fit our pipeline to the data instead of the conventional fit method to incorporate training of models in our pipeline. The fit_transform method is hence used in conjuction with a train set that is being tranformed in the operation as well as used for modelling. The statistics for validation or test data is also calculated in this step and are used when the transform method is called on the pipeline. The pipeline returns the transformed training data and the trained models on calling the fit_transform method. The transform method functions as defined by convention; it transforms the validation/test data.</p>'
st.markdown(summary,unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,6,1])

with col1:
  st.write("")

with col2:
  st.image('app/images/Pipeline.png',caption='Figure 2: End-To-End Pipeline Solution')

with col3:
  st.write("")


level1= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.2rem;">Level 1: Creating Master Data</h3>'
st.markdown(level1,unsafe_allow_html=True) 
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">In level1 we create the master data by combining the files for the six cities. This operation is carried out using pandas concat method with ignoring the indexes of the original datasets so to define a unified index, to disregard the problem of duplicate indices. This ensures that queries on the data set is run smoothly and correctly. This aggregated data is what we are calling the master data. The master data is then split into training and test set which acts an input for level2. The distribution of the dependent variable Price is highly left skewed which makes certain estimates on it very unrelaible, hence we model on the log transformation of the dependent variable. The log transformation brings the distribution closer to a normal distribution and also scales range of the variable. The effect of log transformation are on the dependent variable is shown in Figure 3.</p>'
st.markdown(summary,unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,6,1])

with col1:
  st.write("")

with col2:
  st.image('app/images/PriceDistribution.jpeg',caption='Figure 3: Log transformation on Price')

with col3:
  st.write("")