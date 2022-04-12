import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
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
contributors='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Aamir Ahmad Ansari <a href="https://github.com/aamir09" target="_blank" style="color=#fffff;text-decoration:none">GitHub </a>| <a href="https://www.linkedin.com/in/aamir07/" target="_blank" style="color=#fffff;text-decoration:none">LinkedIn</a></li><ul>'
st.markdown(contributors,unsafe_allow_html=True) 
contributors='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Ibtisam <a href="https://github.com/Ibtisam-Mohammad" target="_blank" style="color=#fffff;text-decoration:none">GitHub </a>| <a href="https://www.linkedin.com/in/ibtim" target="_blank" style="color=#fffff;text-decoration:none">LinkedIn</a></li><ul>'
st.markdown(contributors,unsafe_allow_html=True)
contributors='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Rohit Raj <a href="https://github.com/rraj29" target="_blank" style="color=#fffff;text-decoration:none">GitHub </a>| <a href="https://www.linkedin.com/in/certifieddatascientist/" target="_blank" style="color=#fffff;text-decoration:none">LinkedIn</a></li><ul>'
st.markdown(contributors,unsafe_allow_html=True)  
contributors='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Hemani Shah <a href="https://github.com/hemanishah00" target="_blank" style="color=#fffff;text-decoration:none">GitHub </a>| <a href="https://www.linkedin.com/in/hemani-shah-780a461b2/" target="_blank" style="color=#fffff;text-decoration:none">LinkedIn</a></li><ul>'
st.markdown(contributors,unsafe_allow_html=True) 

### Summary ###
summary_title= '<h2 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.6rem;">Summary</h2>'
st.markdown(summary_title,unsafe_allow_html=True) 
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">In this project, we delve into Indian housing data, looking at the 2011 Indian Housing Census Data as well as house data from Indias metropolite cities. The exploratory data analysis section of the project answers several questions regarding houses in India for the year 2011 and presents some interesting statistics. Predictive modelling of housing prices in Indias metro cities, including Delhi, Mumbai, Bangalore, Chennai, Kolkata, and Hyderabad, is covered in the second section, which is modelling. To compare and pick which model is most suited to our situation, we train linear models, tree-based models, distance-based models, and neural network-based models. We then explain the best model using several approaches.</p>'
st.markdown(summary,unsafe_allow_html=True) 

### EDA ###
eda_title= '<h2 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.6rem;">Exploratory Data Analysis</h2>'
st.markdown(eda_title,unsafe_allow_html=True) 

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">In this section we will explore the Indian Housing Census Data 2011, which is available,<a href="https://censusindia.gov.in/2011census/HLO/HL_PCA/Houselisting-housing-HLPCA.html" target="_blank" style="color=#fffff;text-decoration:none">here</a>. The housing data consist of various features for including state of the condition of houses, materials used in walls and roofs, latrine facilities information, water sources information, basic amenities availibilty and much more. We aim to answer interesting questions about India in 2011 with help of the data set using visualization and explaining their inferences.</p>'
st.markdown(summary,unsafe_allow_html=True) 

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;"><b>Q) Which states in India have good housing conditions and what are some factors affecting this?<b></p>'
st.markdown(summary,unsafe_allow_html=True) 

col1,col2,col3=st.columns([1,8,1])
with col1:
  st.write('')
col2.image('app/images/goodSates.jpeg',caption='States with Good Housing Conditions',width=1200,use_column_width=True)

with col3:
  st.write('')

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">Small Union territories like Daman Diu and Lakshadweep Islands have outclassed big metro cities of India in the year of 2011 for having houses in good condition in per 100 houses. The sizes and population of the Union Territories might be argued with but the fact that they still register larger percentages of houses in good condition is astonishinhg.</p>'
st.markdown(summary,unsafe_allow_html=True) 


summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;"><b>Q) In 2021, According to the reports of Global Health Security Index, India is ranked 68th out of 196 countries in health security. The conditions werent very good in 2011 either, are there any insights that can help us understand the cause?<b></p>'
st.markdown(summary,unsafe_allow_html=True) 

col1,col2,col3=st.columns([1,8,1])
with col1:
  st.write('')
col2.image('app/images/healthIndicators.jpeg',caption='Health Indicators of India',width=1200,use_column_width=True)

with col3:
  st.write('')

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The x-axis represents numbers per 100. On an average than more 25 houses per 100 houses in India has a family size of 6-8 people. Only 20 houses per 100 houses in India have accsess to clean treated water and th rest still rely heavily on sources like Handpumps, uncovered wells, untreated tap water etc. A little less than 70 houses per 100 houses in Inida does not have drinking water sources in their premises. Only 32 house per 100 houses in India have washrooms, the rest rely on littering in public and very few on public washrooms. Almost 60 percent houses does not even have a drainage to treat their waste water, 30 percent have open drainages and only 10 percent have access to closed drainages. Factors like low quality of drinking water, washroom facilities not available in houses, littering in public, no means for waster water to go out and get treated with big sizes of families may be some of the reasons for low health quality of India.</p>'
st.markdown(summary,unsafe_allow_html=True) 

#Q3
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;"><b>Q) In 2011, which states used most eco-friendly and non eco-friendly fuels for cooking?<b></p>'
st.markdown(summary,unsafe_allow_html=True) 

col1,col2,col3=st.columns([2,6,2])
with col1:
  st.write('')
col2.image('app/images/cookingFuel.jpeg',caption='State Wise use of Types Cooking Fuel')

with col3:
  st.write('')

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The green and grey bars represents eco-friendly and non eco-friendly fuels used in different states,the intution here is to show the proportionality in which they were used in 2011. Delhi and Chandigarh used the most eco-friedly cooking fuels while Meghalaya and Jharkhand were on the opposite track. Big states like Maharashtra, Uttar Pradesh, Gujarat and Rajasthan still on majiority use non eco-friendly fuels, which is a concern with respect to environment and the health of the individuals using them.</p>'
st.markdown(summary,unsafe_allow_html=True) 

#Q4

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;"><b>Q) What were the eco-friendly fuels Delhi and Chandigarh used?<b></p>'
st.markdown(summary,unsafe_allow_html=True) 

col1,col2,col3=st.columns([2,6,2])
with col1:
  st.write('')
col2.image('app/images/cookingFuelTop2.jpeg',caption='Types of Fuels Eco-Friendly Fuels Used')

with col3:
  st.write('')

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">It can be observed for both the states that LPG/CNG is the most commonly used fuel for cooking followed by Keroscene. Electricity was used reluctanty by both the states.</p>'
st.markdown(summary,unsafe_allow_html=True) 


#Q5

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;"><b>Q) What were the non eco-friendly fuels Meghalaya and Jharkahand used?<b></p>'
st.markdown(summary,unsafe_allow_html=True) 

col1,col2,col3=st.columns([2,6,2])
with col1:
  st.write('')
col2.image('app/images/cookingFuelbottom2.jpeg',caption='Types of Fuels Non Eco-Friendly Fuels Used')

with col3:
  st.write('')

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">It can be observed for both the states that Firewood is the most commonly used fuel for cooking. Meghalaya seldom used any other fuel while Jharkhand uses variety of them.</p>'
st.markdown(summary,unsafe_allow_html=True) 

#Q6

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;"><b>Q) In which states of India in 2011 does most of Households had the electrnic gadgets?<b></p>'
st.markdown(summary,unsafe_allow_html=True) 

col1,col2,col3=st.columns([2,6,2])
with col1:
  st.write('')
col2.image('app/images/gadgets.jpeg',caption='States With Households Having Electronic Items.')

with col3:
  st.write('')

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The observations tell that in NCT of Delhi more than 80 percent households had electronic items/gadgets at home, followed closely by Tamil Nadu and Kerala. Jharkhand again at the bottom of yet another metric , accompained by Bihar. </p>'
st.markdown(summary,unsafe_allow_html=True) 

#Q7
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;"><b>Q) Distribution of Share of Permanent Houses between Rural and Urban India in 2011?<b></p>'
st.markdown(summary,unsafe_allow_html=True) 

col1,col2,col3=st.columns([3,4,3])
with col1:
  st.write('')
col2.image('app/images/PeramanentHouses.jpeg',caption='Share of Permanent Houses')

with col3:
  st.write('')

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">Rural India had almost half as many permanent houses as in Urban India in 2011.  </p>'
st.markdown(summary,unsafe_allow_html=True) 

#Q8 

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;"><b>Q) What types of latrine flushes were popular in households of different states of India?<b></p>'
st.markdown(summary,unsafe_allow_html=True) 

col1,col2,col3=st.columns([2,6,2])
with col1:
  st.write('')
col2.image('app/images/laterineFlushes.jpeg',caption='Laterine Flush Systems Used in India')

with col3:
  st.write('')

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">Piped sewage system looks like a luxuxry only available at capacity in Chandigarh and New Delhi. Rest of India still uses Septic Tanks for onsite waste water disposal.</p>'
st.markdown(summary,unsafe_allow_html=True) 

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">More interesting questions are answered in the notebook available at <a href="https://github.com/aamir09/DS2Project" target="_blank" style="color=#fffff;text-decoration:none">GitHub</a>.</p>'
st.markdown(summary,unsafe_allow_html=True) 

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
  st.image('app/images/Pipeline.jpeg',caption='Figure 2: End-To-End Pipeline Solution')

with col3:
  st.write("")

with col1:
  st.write("")

with col2:
  st.image('app/images/pipelineWorkflow.jpeg',caption='Figure 3: Workflow')

with col3:
  st.write("")

### Level 1 ####
level1= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.2rem;">Level 1: Creating Master Data</h3>'
st.markdown(level1,unsafe_allow_html=True) 
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">In level1 we create the master data by combining the files for the six cities. This operation is carried out using pandas concat method with ignoring the indexes of the original datasets so to define a unified index, to disregard the problem of duplicate indices. This ensures that queries on the data set is run smoothly and correctly. This aggregated data is what we are calling the master data. The master data is then split into training and test set which acts an input for level2. The distribution of the dependent variable Price is highly left skewed which makes certain estimates on it very unrelaible, hence we model on the log transformation of the dependent variable. The log transformation brings the distribution closer to a normal distribution and also scales range of the variable. The effect of log transformation are on the dependent variable is shown in Figure 4.</p>'
st.markdown(summary,unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,6,1])

with col1:
  st.write("")

with col2:
  st.image('app/images/PriceDistribution.jpeg',caption='Figure 4: Log transformation on Price')

with col3:
  st.write("")

#### Level 2 ####
level2= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.2rem;">Level 2: Cleaning Pipeline</h3>'
st.markdown(level2,unsafe_allow_html=True) 

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The cleaning pipeline is the first part of the end-to-end training pipeline, it not only cleanes the data, it transforms our data and add new features to our data(feature engineering). The follwing operations are performed by the cleaning pipeline:</p>'
st.markdown(summary,unsafe_allow_html=True)

first='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Drops Unecessary Columns: </b>The drop columns operation is a custom class in our pipeline which takes a list of columns and drop them from the dataset. As majiority columns had more than 70% null values, we did not drop on the percentage null values in a column. Generally the threshold of dropping columns in percentage null values is between 50-80 percent.</li><ul>'
st.markdown(first,unsafe_allow_html=True) 

second='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Replace NaN Values: </b>The null values in the binary features is represented by the digit 9, this operation finds those values and replace them with Nan(Not a Number). Nan is a special floating point number which is used as a standard to represent missing values.</li><ul>'
st.markdown(second,unsafe_allow_html=True) 

third='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Add Area Bins: </b>This is one of the feature engineering operations that we do in our pipeline. It evalutes the quantiles of the feature Area and assigns the corresponding quantile number(1, 2, 3, 4) in which a sample resides. This provides a relative information of the size of a house and we call this new variable AreaType; small if it resides in quantile 1 encoded as 1, medium if it resides in quantile 2 encoded as 2, large if it resides in quantile 3 encoded as 3 and very large if it resides in quantile 4 encoded as 4.</li><ul>'
st.markdown(third,unsafe_allow_html=True) 

fourth='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Custom Imputer: </b>This is one of the most important operation of this pipeline, imputing the null/NaN values. We came up with a better solution than just imputing the null values with most frequent classes in the in dataset for each feature. Our custom imputer uses the relative area information using the area bins created earlier. For each feature it calculates the most frequent class for each area type(1, 2, 3, 4)  and then imputes the null values residing in a particular area type with their respective calculated modes. This operation results in more localized imputation and less information loss.</li><ul>'
st.markdown(fourth,unsafe_allow_html=True) 

fifth='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Handle Outliers: </b>In this operation we handle the outliers in feature Area as shown above on the right of Figure 1 by replacing the outliers with by Q1-1.5*IQR and Q3+1.5*IQR, where Q1, Q2 and IQR is the 25th percentile, 75th percentile and Interquartile range respectively; any things less than Q1-1.5*IQR is reassigned as Q1-1.5*IQR and any value greater than Q3+1.5*IQR to Q3+1.5*IQR.</li><ul>'
st.markdown(fifth,unsafe_allow_html=True) 

sixth='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>One Hot Encoder: </b>One hot encoding is extensively used in encoding categorical columns to series of binary dummy columns, one for each category in which 0 represents that a sample does not correspond to the class where as 1 represent that a sample represent to this particular class. We used one hot encoding to convert our city variable conatining city names to binary dummy variables so that the machine can process this information.</li><ul>'
st.markdown(sixth,unsafe_allow_html=True) 

seventh='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Feature Engineering: </b>Though we have done some feature engineering above but this is the more advanced one as we augment our data with the Indian Housing Census Data 2011. Our aim is to add  Housing Quality of Living Index(HQLI) for each state in our dataset. It depends upon three variables which are quality of housing index(QHI), basic amenity index(BAI), asset index(AI). HQLI = QHI + BAI + AI. The feature information to be cosidered for calculation of QHI are:\n<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">HouseHolds by good condition of residential census houses</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">HouseHolds living in permanent houses</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Married couples do not have exclusive room</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">HouseHolds with own houses</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">HouseHolds having at least two dwelling rooms</li></ul>\nBAI are: \n<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Drinking water with in premises</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Electricity</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Latrine within premises</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Bath room</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Closed drainage system for waste water outlet</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Separate kitchen inside the house</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">LPG/PNG for cooking</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Banking service</li></ul>\nAI are: \n<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Radio/Transistor</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Television</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Telephone facilities (mobile, landline or both)</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Bicycle</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Scooter/Motorcycle/Moped</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Car/Jeep/Van</li><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;">Computer/Laptop (with or without internet)</li></ul>These features are aggreagted by state name and then a weighted(randomly intialised weights) sum is calcultaed for QHI, BAI, AI(as shown below) each state. These 3 weighted sums for each state is then added to get HQLI for each state.</li></ul>'
st.markdown(seventh,unsafe_allow_html=True) 
st.latex('QHI\ =w_1\cdot f_1+w_2\cdot f_2+w_3\cdot f_3+w_4\cdot f_4+w_5\cdot f_5')
st.latex('BAI\ =w_1\cdot f_1+w_2\cdot f_2+w_3\cdot f_3+w_4\cdot f_4+w_5\cdot f_5+w_6\cdot f_6 + w_7\cdot f_7 + w_8\cdot f_8 ')
st.latex('AI\ =w_1\cdot f_1+w_2\cdot f_2+w_3\cdot f_3+w_4\cdot f_4+w_5\cdot f_5+w_6\cdot f_6 + w_7\cdot f_7')
st.latex('HQLI = QHI + BAI +AI')
st.latex('where\ w_i\ are\ the\ randomly\ initialized\ weights\, f_i\ are\ the\ feratures\ to\ be\ used.')

eigth='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Standardization:</b> We standardize our continious features using this operation, Standardization comes into picture when features of input data set have large differences between their ranges, or simply when they are measured in different measurement units. We take down the mean of the distribution to zero and the standard deviation to 1, stabalizing the variance and the learning process.[1]</b> </li></ul>'
st.markdown(eigth,unsafe_allow_html=True) 


### Level 3 ###
level3= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.2rem;">Level 3: Modelling</h3>'
st.markdown(level3,unsafe_allow_html=True) 
summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">Level 3 is the second and the last part of our end-to-end pipeline, where we train various machine learning and deep learning models. The transformed training set is taken from the cleaning pipeline and passed on to several regression models. The complete pipeline of level1 and level2 returns these models with the training data. We built 7 different regression models for our problem, from linear regression being the simplest to the most complex neural networks. Each models hyper paramerters were tuned using 3-fold cross validation using GridSearchCV. The best estimators for each model is saved and a performance matrix is also calculated which consist of r2 score and mse on train and test datasets. The following models were built during the training process:</p>'
st.markdown(summary,unsafe_allow_html=True)

first='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Linear Regression: </b>Linear Regression is a predictive modelling technique in which you model a dependent variable <i>y</i>  on an independent variable <i>X</i>, where x can be a single feature or a vector of features. The aim is to find a best fit line to the available data. The assumptions of the algorithm are simple, there should be a linear relationship between predictor(dependent variable) and the features(independent variables), Homoscedasticity, observations are independent of each other, the residuals are normally distributed.</li><ul>'
st.markdown(first,unsafe_allow_html=True) 
st.latex('y\ =beta\cdot X')

second='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>K-Nearest Neighbors: </b>KNN regression is a non-parametric method that, in an intuitive manner, approximates the association between independent variables and the continuous outcome by averaging the observations in the same neighbourhood. The size of the neighbourhood needs to be set by the analyst or can be chosen using cross-validation to select the size that minimises the mean-squared error.[2]</li><ul>'
st.markdown(second,unsafe_allow_html=True) 

third='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Neural Network: </b>Neural networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data. As such, they tend to resemble the connections of neurons and synapses found in the brain. Neural networks with several process layers are known as "deep" networks and are used for deep learning algorithms.[3] We use six layers with varying neurons coupled with LeakyRelu activation. To prevent us from over fitting a precautionary early stopping callback is inserted which monitors the validation loss and if no improvement is observed inside a patience level, the training stops and best weights of the model are restored. To optimze our mse loss plain we are using Adam optimizer. </li><ul>'
st.markdown(third,unsafe_allow_html=True) 

fourth='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Decision Tree: </b>Decision Trees are a type of Supervised Machine Learning (that is you explain what the input is and what the corresponding output is in the training data) where the data is continuously split according to a certain parameter. The tree can be explained by two entities, namely decision nodes and leaves. The leaves are the decisions or the final outcomes. And the decision nodes are where the data is split.[4] For regression the nodes are split upon the largest variance reduction and the predictions are the aggregated mean of the values of the leaf node in which the sample resides. The decision trees are built in a depth first manner from top to bottom and the complexity of the model increases as the depth of the tree increases, In our case the optimal depth of the decision tree after 3-fold cross validation came out to be 9.</li><ul>'
st.markdown(fourth,unsafe_allow_html=True) 

fifth='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Bagging Ensembles: </b>Bagging is an ensemble based machine learning algorithm based on descision trees that creates an ensemble of strong learners and take the average of the output all the trees in the emsemble as an output in a an effort to decrease the variance of the predictions. Complex decision trees are prone to overfitting, a small change in the distribution of test data will become a cause of great fluctuation in the output and bagging tries to combat this problem.</li><ul>'
st.markdown(fifth,unsafe_allow_html=True) 

sixth='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Random Forest: </b>Random Forest is a similar algorithm like bagging, it creates an ensemble of strong learners fitted on different bootsrapped sample of the training data but the learners here are decorrelated with each other(only a randomly chosen fraction of the features are given to each node to configure its split). This lack of correlation help us lower down the variance of the predictions further more.</li><ul>'
st.markdown(sixth,unsafe_allow_html=True) 

seventh='<ul><li style="font-family:Courier; color:#fff; align:left;font-size: 1.2rem;"><b>Gradient Boosting: </b>In gradient boosting machines, or simply, GBMs, the learning procedure consecutively fits new models to provide a more accurate estimate of the response variable. The principle idea behind this algorithm is to construct the new base-learners to be maximally correlated with the negative gradient of the loss function, associated with the whole ensemble. The loss functions applied can be arbitrary, but to give a better intuition, if the error function is the classic squared-error loss, the learning procedure would result in consecutive error-fitting. In general, the choice of the loss function is up to the researcher, with both a rich variety of loss functions derived so far and with the possibility of implementing ones own task-specific loss.[5]</li><ul>'
st.markdown(seventh,unsafe_allow_html=True) 

##### Model Comparison ##########
comparison= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.4rem;">Model Comparison</h3>'
st.markdown(comparison,unsafe_allow_html=True) 

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">In this section we compare all machine learning models trained for our problem on the basis of their performance on train and test data using various metrics and methods.</p>'
st.markdown(summary,unsafe_allow_html=True)

mse= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.2rem;">Mean Squared Error</h3>'
st.markdown(mse,unsafe_allow_html=True) 

performance=None
with open('PartB/models/performanceMatrix.pickle','rb') as f:
  performance=pickle.load(f)

col1, col2= st.columns([5,5])

mseTuple=[(performance[i]['MSE']['train'],performance[i]['MSE']['test']) for i in performance]
nameMapping={'nn':'Neural Network','lreg':'Linear Regression','cboost':'CatBoost','tree':'Decision Tree','forest':'Random Forest','knn':'KNN','gboost':'Gradient Boosting','bagging':'Bagging Ensemble'}
names=[nameMapping[i] for i in performance]
train,test=zip(*mseTuple)

mseDf=pd.DataFrame()
mseDf['names']=names
mseDf['train']=train
mseDf['test']=test

with col2:
  fig,ax=plt.subplots()
  diff=mseDf.set_index('names')['train']-mseDf.set_index('names')['test']
  np.abs(diff.sort_values()).plot.barh(ax=ax,width=0.4)
  ax.set_title('Generalization Error',fontsize=20)
  ax.set_xlabel('Error',fontsize=15)
  ax.set_ylabel('Models',fontsize=15)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  st.pyplot(fig,height=200)
with col1:
  fig,ax=plt.subplots()
  mseDf.set_index('names').sort_values(['test','train']).plot.barh(ax=ax,width=0.4)
  ax.set_title('Performance on Test & Train Set',fontsize=20)
  ax.set_xlabel('MSE',fontsize=15)
  ax.set_ylabel('Models',fontsize=15)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  st.pyplot(fig,height=200)



summary=f'<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">Random forest outperforms every other model in terms of test mse while the most complex network in the kitty; the neural network has the worst mse on the test set. The above graph is sorted with respect to the test set mse. The generalization error is the difference between the mse of train set and test set. Random forest seems to generalise poorly as compared to other models with higher mse.</p>'
st.markdown(summary,unsafe_allow_html=True)

r2score= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.2rem;">R2 Score</h3>'
st.markdown(r2score,unsafe_allow_html=True) 

r2Tuple=[(performance[i]['R2']['train'],performance[i]['R2']['test']) for i in performance]
trainr,testr=zip(*r2Tuple)
r2Df=pd.DataFrame()
r2Df['names']=names
r2Df['train']=trainr
r2Df['test']=testr

col1, col2= st.columns([5,5])
with col1:
  fig,ax=plt.subplots()
  r2Df.set_index('names').sort_values(['test','train']).plot.barh(ax=ax,width=0.4)
  ax.set_title('Performance on Test & Train Set',fontsize=20)
  ax.set_xlabel('R2 Score',fontsize=15)
  ax.set_ylabel('Models',fontsize=15)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  st.pyplot(fig,use_column_width=True)

with col2:
  fig,ax=plt.subplots()
  diff=r2Df.set_index('names')['train']-r2Df.set_index('names')['test']
  np.abs(diff.sort_values()).plot.barh(ax=ax,width=0.4)
  ax.set_title('Generalization Error',fontsize=20)
  ax.set_xlabel('Error',fontsize=15)
  ax.set_ylabel('Models',fontsize=15)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  st.pyplot(fig,use_column_width=True)



summary=f'<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">Neural Network again the worst performer and is not able to explain any variance of the features in the predictions. The Random Forest does bag the highest rank but again the generalization error is quite high, in comparison, bagging ensemble looks more robust as the generalization is low and test mse and r2score are similar to that of Random Forest. Since the bagging model is more robust we choose the bagging model as our best performing model.</p>'
st.markdown(summary,unsafe_allow_html=True)

#### Understanding the Model #### 
understanding= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.4rem;">Understanding the Model</h3>'
st.markdown(understanding,unsafe_allow_html=True) 

summary=f'<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The understanding of machine learning model has a great importance in real world. If model classifes an image correctly then what are the features that is considering important in the image to generate such predictions? IF you are working in a company solving business problems, the stake holder would want to know what makes your model provide that solution. While worling in a medical field for instance in a disease detection system, you will be asked how your model is detecting the disease. Hence understsanding and explaining your model is a cruical part of being a data scientist and a machine learning practitioners. Another advantage of knwoing your model is that you can debug your model. If you know what is going wrong then probabilities are high that you can fix it, right?</p>'
st.markdown(summary,unsafe_allow_html=True)

summary=f'<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The simplest model to understand is the Linear Regression model because the weights directly has a say in the impact the features make but in complex models like neural networks and tree ensembles it might be a little bit more difficult to interpret them and these models are called as black box models. Our best model happens to be a tree ensemble; the bagging tree model. We will globally inspect this model by retrieving the feature importaces that will idicate what ticks our model globally and which features affect the most in the decision making. We will then dive into a local inspection of the predictions with the most error and the least error and get know what features played a role in this process.</p>'
st.markdown(summary,unsafe_allow_html=True)

globalImp= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.2rem;">Global Feature Importance</h3>'
st.markdown(globalImp,unsafe_allow_html=True) 

col1, col2, col3= st.columns([3,4,3])

with col1:
  st.write('')

with col2:
  st.image('app/images/featureImportanceGlobal.jpeg','Figure 5: Global Feature Importance')

with col3:
  st.write('')

summary=f'<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">Figure 5 conveys the feature importance calculated as the average decrease in variance in the bagging ensemble shown in green and permuation feature importance calculated by permuting a feature column at once and noticing the drop in the oberving metrics shown in grey. These are the top 12 features who contributes gloabally in a predictions, the impact of the rest are negligible. Area is an expected feature to pop up in such metrics and has been ranked the top most feature that contributes globally in making predictions. It is interesting to see that onehot encoded variables for cities are considered importance as well which do indicate that the model is learning from the trend of prices in each city. Number of bedrooms has also made it place and it makes sense as that the price a house with more bedroom is more than a house with less number of bedrooms. Resale, availability of car parking and intercom are also key factors. The engineered feature HQLI has a relatively higher importance to the 30 or more features below it. Rain water harvesting is interesting to see in the mix.</p>'
st.markdown(summary,unsafe_allow_html=True) 

localImp= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.2rem;">Local Feature Importance</h3>'
st.markdown(localImp,unsafe_allow_html=True) 

summary=f'<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">We will asses the individual instances, an instance with lowest squared error and an instance with highest squared error and inspect what features affected their results. To make this investigation happen we will use LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS(LIME) algorithm that is a technique that approximates any black box machine learning model with a local, interpretable model to explain each individual prediction.[6]</p>'
st.markdown(summary,unsafe_allow_html=True)

summary=f'<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">Lets examine the prediction with the least squared error and the most squared error. Left: The sample is of the test set at the index 2758, the true value at log scale is 15.75 and having the least squared error. Right:  The sample is of the test set at the index 2917, the true value at log scale is 19.20, having the greatest squared error. </p>'
st.markdown(summary,unsafe_allow_html=True)

with open('PartB/models/savedModels/leastExplanation.pickle','rb') as f:
  minHtml=pickle.load(f)

with open('PartB/models/savedModels/mostExplanation.pickle','rb') as f:
  mostHtml=pickle.load(f)

col1,col2, col3= st.columns([5,1,5])

test=pd.read_csv('test.csv').drop('Unnamed: 0',axis=1)
sample1=test.iloc[2758,:]
sample2=test.iloc[2917,:]

with col1:
  st.dataframe(sample1,500,100)
  st.write('')
  white_background = "<style>:root {background-color: #ffffff}</style>"
  st.components.v1.html(white_background + minHtml,width=800,height=500)

with col2:
  st.write('')

with col3:
  st.dataframe(sample2,500,100)
  st.write('')
  white_background = "<style>:root {background-color: #ffffff}</style>"
  st.components.v1.html(white_background + mostHtml,width=800,height=500)

summary=f'<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">The interpretation for Left: The location is in Bangalore and not Mumbai hence Mumbai being 0 will decrease the final output score(Mumbai is expensive than other cities hence it makes sense), the location is not Kolkata which add up in the final predictions, simiarly positive weighted variables adds to the final prediction score while negatively weighted ones, decrease it. Right: The sample not being from Mumbai drops it Price as well as having less area. What works in its favor is that it is from Delhi and not from Kolkata but the price is still off by alot. The binary variable here is holding the key as we saw in even the most accurate one on the left. The permuataion importance as well as feature importance also gave a similar account for Mumbai variable and now we also know how much impact and in which direction it may have on our predictions.</p>'
st.markdown(summary,unsafe_allow_html=True)

##### References #######

references= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.4rem;">References</h3>'
st.markdown(references,unsafe_allow_html=True) 

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">[1] When and Why to Standardize Your Data?, https://builtin.com/data-science/when-and-why-standardize-your-data</p>'
st.markdown(summary,unsafe_allow_html=True)

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">[2] K-nearest Neighbours Regression, https://bookdown.org/tpinto_home/Regression-and-Classification/k-nearest-neighbours-regression.html</p>'
st.markdown(summary,unsafe_allow_html=True)

summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">[3] Neural Network, https://www.investopedia.com/terms/n/neuralnetwork.asp</p>'
st.markdown(summary,unsafe_allow_html=True)

summary='<p style="font-family:Courier;text-align:left; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">[4] Decision Trees for Classification: A Machine Learning Algorithm, https://www.xoriant.com/blog/product-engineering/decision-trees-machine-learning-algorithm.html</p>'
st.markdown(summary,unsafe_allow_html=True)

summary='<p style="font-family:Courier;text-align:left; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">[5] Gradient boosting machines, a tutorial, https://www.frontiersin.org/articles/10.3389/fnbot.2013.00021/full</p>'
st.markdown(summary,unsafe_allow_html=True)

summary='<p style="font-family:Courier;text-align:left; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">[6] What is Local Interpretable Model-Agnostic Explanations (LIME)?, a tutorial, https://c3.ai/glossary/data-science/lime-local-interpretable-model-agnostic-explanations/</p>'
st.markdown(summary,unsafe_allow_html=True)