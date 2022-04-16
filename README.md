# Deep Dive of Housing in India 

## Part A - Comprehensive Analysis of India Housing Census Data 2011

<p align="center">
  <img 
    width="400"
    height="300"
    src="https://user-images.githubusercontent.com/62461730/163445913-ccaea730-6521-4633-9669-6b7076e848ed.png"
  >
</p>

-- We fond a negative correlation between Households using electricity and those using Kerosene - which implies less reliance on Kerosene as electricity distribution increases

 <p align="center">
  <img 
    width="800"
    height="300"
    src="https://user-images.githubusercontent.com/62461730/163446122-69b7efe9-e990-48ee-b28c-a68ddc5c3f27.png"
  >
</p>

-- The union territories had much better kitchen facilities compared to the other states of India and India as a whole. 

<p align="center">
  <img 
    width="600"
    height="400"
    src="https://user-images.githubusercontent.com/62461730/163446279-7a713289-c97d-4528-9ff9-606672ad6edb.png"
  >
</p>


-- While comparing each facility of a state against the country average, some states like Chandigarh, have much better water facilities than India as a whole.
 


 ![image](https://user-images.githubusercontent.com/62461730/163446474-ae7e4188-480a-401f-ba6e-93715b3deca5.png)

On Comparing the distribution of the facility across the country,  Bihar, Assam, Jharkhand, Meghalaya, Uttar Pradesh and Odisha heavily rely on Kerosene.
 

•	Created a filtering algorithm which basically answers the question:
 What state should I live in?
o	if the user wants to live in a house with the following conditions :
	For example: Assets like internet connection is very important to the user so they set a threshold of 50%. This means that our algorithm will filter out all the localities in which less then 50% of the houses have internet connection.
o	Once the user decides on all the fields, we will try to find a locality where such houses are typical.
o	Finally we plot the count of localities per state which tell us which states will have high probabilities of having houses like the one the user describes(implemented in filtering_final.ipynb)

![image](https://user-images.githubusercontent.com/62461730/163446727-9b311336-ec2a-4f75-9d1a-1efd0be5d50c.png)

 
o	We have implemented this in our UI in more detail: we create a plot by taking the mean of percentage of the features that the user selects for a particular field for all the states.
 
![image](https://user-images.githubusercontent.com/62461730/163446831-c70e2610-6bb8-4069-928a-1b83798155f0.png)



## Part B - Housing Price Prediction from Metropolitan Indian Cities

This dataset comprises data that was scraped. It includes:
•	collection of prices of new and resale houses located in the metropolitan areas of India
•	the amenities provided for each house
Our main.py contains the general pipeline used for our solution:

![image](https://user-images.githubusercontent.com/62461730/163243136-c918b192-f0d3-4876-9e1d-696e965a40fd.png)

 
### Level 1: Creating Master Data: combining the files for the six cities(using functions defined in CreateMasterDataset file)

### Level 2: Cleaning Pipeline(functions defined in pipelineClasses file): 

•	Drops Unecessary Columns

•	Replace NaN Values

•	Add Area Bins

•	Custom Imputer

•	Handle Outliers

•	One Hot Encoder

•	Feature Engineering: Added HQLI

•	Standardization

### Level 3: Modelling

•	Linear Regression

•	K-Nearest Neighbors

•	Neural Network

•	Decision Tree

•	Bagging Ensembles

•	Random Forest

•	Gradient Boosting

Code for all the models have been saved in the folder Part B/models. The model weights have been saved in PartB/models/savedModels.
We then find the mse and R2 scores of each model and compare them.
We also visualized local and global feature importance to understand the model better. This code can be found in eda and eda2 files.

## References:
https://www.youtube.com/watch?v=YCwRd-N3D14

