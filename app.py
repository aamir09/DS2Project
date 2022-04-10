import streamlit as st
import numpy as np
import matplotlib.pyplot as plt# matplotlib.inline
import plotly as pt
import pandas as pd
import wget
import gdown
import os
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Indian Housing Data",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://github.com/aamir09/DS2Project',
         'Report a bug': "https://github.com/aamir09/DS2Project",
         'About': "# This is a header. This is an *extremely* cool app!"
         })
rad = st.sidebar.radio('Navigation',['Home','Search'])

path='E:/Projects/DS2Project/app/df_exp/'
files=os.listdir(path)

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def ind(i):
  return pd.read_csv(path+i).drop(['State Code','District Code'],axis=1)
for i in files:
  nam=i.split('.')[0]
  globals()[nam]=ind(i)



if rad =='Search':

  dic_={
  "Ameneties":df_assets,
  "Bathroom type":df_bath,
  "Condition of house":df_condn,
  "Couple":df_couple,
  "Material of Floor":df_floor,
  "Fuel for cooking":df_fuel,
  "Household size":df_hhold,
  "House type":df_house,
  "Kitchen":df_kitchen,
  "Lighting source":df_light,
  "Ownership status":df_owner,
  "Material of Roof":df_roof,
  "No of Rooms":df_room,
  "Toilet facillities":df_toilet,
  "Material of Wall":df_wall,
  "Waste Outlet":df_waste,
  "Water source and location":df_water
  }
  st.markdown(
  """
  <style>
  [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {width: 200px;}
  </style>
  """,
  unsafe_allow_html=True,)
  @st.cache(allow_output_mutation=True,suppress_st_warning=True)
  def fil():
    B=pd.read_csv('E:/Projects/DS2Project/app/clean_file.csv')
    return B
  B=fil()

  m1=st.multiselect('Facilities to choose',dic_.keys())
  if m1:
    st.markdown('<hr>',True)
  
  m_lst=[]
  col1,col4=st.columns([1,8])
  for i in m1:
    A=dic_[i]
    m1_A=st.multiselect(f'Choose further from {i}',A.columns[2:])
    m_lst.append(m1_A)
  m_lst_f=sum(m_lst, [])

  
  slider = st.slider('Number of data entries to show', min_value=0,
                    max_value=100, value= 10, key='my_slider')
  if 'k1' not in st.session_state:
    st.session_state.k1=7
  if 'k2' not in st.session_state:
    st.session_state.k2=10
  width = st.slider("Plot width", 1, 25, 3,key='k1')
  height = st.slider("Plot height", 1, 25, 1,key='k2')
  
  rad_a=st.radio('Show by:',['State','City'])
  if rad_a=='State':

    temp_=B.groupby('State Name').mean()[m_lst_f]
    temp_['sum']=np.sum(temp_,axis=1).values
    if st.button("Filter") :
      st.markdown('<hr>',True)
      fig1, ax = plt.subplots(figsize=(width, height))
      temp_.sort_values('sum',ascending=False)[:slider].iloc[:,:-1].plot(kind='barh',colormap='Set1',
                                    stacked=True,legend=False,ax=ax)
      st.pyplot(fig1)

  if rad_a=='City':
    temp_=B.groupby('District Name').mean()[m_lst_f]
    temp_['sum']=np.sum(temp_,axis=1).values
    if st.button("Filter") :
      st.markdown('<hr>',True)
      fig2, ax = plt.subplots(figsize=(width, height))
      temp_.sort_values('sum',ascending=False)[:slider].iloc[:,:-1].plot(kind='barh',colormap='Set1',
                                    stacked=True,legend=False,ax=ax)
      st.pyplot(fig2)
  


  







if rad=='Home':
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



  ### Level 3 ###
  level3= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.2rem;">Level 3: Modelling</h3>'
  st.markdown(level3,unsafe_allow_html=True) 
  summary='<p style="font-family:Courier;text-align:justify; weight:bold;color:#ffffff; align:left;font-size: 1.2rem;">Level 3 is the second and the last part of our end-to-end pipeline, where we train various machine learning and deep learning models. The transformed training set is taken from the cleaning pipeline and passed on to several regression models. The complete pipeline of level1 and level2 returns these models with the training data.</p>'
  st.markdown(summary,unsafe_allow_html=True)


  ##### Model Comparison ##########
  comparison= '<h3 style="font-family:Courier; weight:bold;color:#FA8072; align:left;font-size: 1.4rem;">Model Comparison</h3>'
  st.markdown(comparison,unsafe_allow_html=True) 
