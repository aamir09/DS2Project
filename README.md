# Deep Dive of Housing in India 

## Part A - Comprehensive Analysis of India Housing Census Data 2011

### We fond a negative correlation between Households using electricity and those using Kerosene - which implies less reliance on Kerosene as electricity distribution increases.

<p align="center">
  <img 
    width="500"
    height="500"
    src="https://user-images.githubusercontent.com/62461730/163445913-ccaea730-6521-4633-9669-6b7076e848ed.png"
  >
</p>

### The union territories had much better kitchen facilities compared to the other states of India and India as a whole. 

 <p align="center">
  <img 
    width="800"
    height="300"
    src="https://user-images.githubusercontent.com/62461730/163684395-e861c213-f6f0-44dc-bd1e-52408b505dfa.jpg"
  >
</p>


### While comparing each facility of a state against the country average, some states like Chandigarh, have much better water facilities than India as a whole.

![CHANDIGARHwater](https://user-images.githubusercontent.com/62461730/163684471-aaacc5b7-50c3-4a41-9d73-64f19838cee5.jpg)

### On Comparing the distribution of the facility across the country,  Bihar, Assam, Jharkhand, Meghalaya, Uttar Pradesh and Odisha heavily rely on Kerosene.

 ![kerosene_1 (1)](https://user-images.githubusercontent.com/62461730/163684441-0c643364-dd26-4e3e-a5c1-572e3fa4a20a.jpg)


### Filtering 

Two filtering algorithms are built to answer a question <b>What state should I live in?</b>. The principle of both the algorihtms are same, filter the states on the basis of their basic amenities and characteristics and provvide end user the results. The difference is in the results algorithm 1 gives us the count of districts in state where there are availibiity of certain amenity satisfies the user defined threshold. Algorithm 2 provides us with visualizatios of mean percentage availibility of the assets selected by the users. Algorithm 2 can be observed in the UI and Algorithm 1 can be found in the PART A folder.
 

## Part B - Housing Price Prediction from Metropolitan Indian Cities

The data set is an open source data set available at kaggle. The data contains information about the house and its amenities as mentioned above. The data is distributed in 5 files and each represent a metro city. Delhi, Mumbai, Bangalore, Chennai, Hyderabad and Kolkata are the ones that are present in the data. Price is our target variable and the rest 39 variables are accounted as feature variables.

### APPROACH A

![image](https://user-images.githubusercontent.com/62461730/163243136-c918b192-f0d3-4876-9e1d-696e965a40fd.png)

 
### Level 1: Creating Master Data

In level1 we create the master data by combining the files for the six cities. This operation is carried out using pandas concat method with ignoring the indexes of the original datasets so to define a unified index, to disregard the problem of duplicate indices. This ensures that queries on the data set is run smoothly and correctly. This aggregated data is what we are calling the master data. The master data is then split into training and test set which acts an input for level2. The distribution of the dependent variable Price is highly left skewed which makes certain estimates on it very unrelaible, hence we model on the log transformation of the dependent variable. The log transformation brings the distribution closer to a normal distribution and also scales range of the variable. The effect of log transformation are on the dependent variable is shown below.

![image](https://user-images.githubusercontent.com/62461730/163683768-a5a64360-cbd8-4c8c-90ae-4ff457967550.png)


### Level 2: Cleaning Pipeline(functions defined in pipelineClasses file): 

•	Drops Unecessary Columns: The drop columns operation is a custom class in our pipeline which takes a list of columns and drop them from the dataset.

•	Replace NaN Values: The null values in the binary features is represented by the digit 9, this operation finds those values and replace them with Nan(Not a Number)

•	Add Area Bins: This is one of the feature engineering operations that we do in our pipeline. It evalutes the quantiles of the feature Area and assigns the corresponding quantile number(1, 2, 3, 4) in which a sample resides. This provides a relative information of the size of a house and we call this new variable AreaType; small if it resides in quantile 1 encoded as 1, medium if it resides in quantile 2 encoded as 2, large if it resides in quantile 3 encoded as 3 and very large if it resides in quantile 4 encoded as 4.

•	Custom Imputer: This is one of the most important operation of this pipeline, imputing the null/NaN values. We came up with a better solution than just imputing the null values with most frequent classes in the in dataset for each feature. Our custom imputer uses the relative area information using the area bins created earlier.

•	Handle Outliers: In this operation we handle the outliers in feature Area as shown above on the right of Figure 1 by replacing the outliers with by Q1-1.5*IQR and Q3+1.5*IQR, where Q1, Q2 and IQR is the 25th percentile, 75th percentile and Interquartile range respectively; any things less than Q1-1.5*IQR is reassigned as Q1-1.5*IQR and any value greater than Q3+1.5*IQR to Q3+1.5*IQR.

•	One Hot Encoder: One hot encoding is extensively used in encoding categorical columns to series of binary dummy columns, one for each category in which 0 represents that a sample does not correspond to the class where as 1 represent that a sample represent to this particular class.

•	Feature Engineering: Though we have done some feature engineering above but this is the more advanced one as we augment our data with the Indian Housing Census Data 2011. Our aim is to add Housing Quality of Living Index(HQLI) for each state in our dataset. It depends upon three variables which are quality of housing index(QHI), basic amenity index(BAI), asset index(AI). HQLI = QHI + BAI + AI.

•	Standardization: We standardize our continious features using this operation, Standardization comes into picture when features of input data set have large differences between their ranges, or simply when they are measured in different measurement units. We take down the mean of the distribution to zero and the standard deviation to 1, stabalizing the variance and the learning process

### Level 3: Modelling

•	Linear Regression

•	K-Nearest Neighbors

•	Neural Network

•	Decision Tree

•	Bagging Ensembles

•	Random Forest

•	Gradient Boosting Algorithms

### APPROACH B

<p align="center">
  <img 
    width="1000"
    height="400"
    src="https://user-images.githubusercontent.com/62461730/163683664-82c93887-4f76-424c-af1c-7dbb3d8aa6fa.png"
  >
</p>


Through this model we ensure that the highest importance is given to the city in which the house is located in. From our analysis we know that the highest importance is given to features which represent city like mumbai, kolkata, etc. This model also considers the fact that a house cannot be in 2 cities at once. for example, any record cannot contain Bangalore=1 and Chennai=1 as well (unless it is a mistake in which case the decision tree will traverse to sub-tree under the 'Other' branch).
So in this approach we divided the training data according to city. We will then train 6 decision trees for all the cities present in the training data. We also train a 7th decision tree on the whole training dataset so that if the test dataset contains a record of a house in a new city(city which is not in training) then the model can still predict for that record. We then aggregated all our results into a list and then found our mean squared error and R2 score for train data as well as test data.





## References:
https://www.youtube.com/watch?v=YCwRd-N3D14

