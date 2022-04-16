# Deep Dive of Housing in India 

## Part A - Comprehensive Analysis of India Housing Census Data 2011

### We fond a negative correlation between Households using electricity and those using Kerosene - which implies less reliance on Kerosene as electricity distribution increases.

<p align="center">
  <img 
    width="400"
    height="300"
    src="https://user-images.githubusercontent.com/62461730/163445913-ccaea730-6521-4633-9669-6b7076e848ed.png"
  >
</p>

### The union territories had much better kitchen facilities compared to the other states of India and India as a whole. 

 <p align="center">
  <img 
    width="800"
    height="300"
    src="https://user-images.githubusercontent.com/62461730/163446122-69b7efe9-e990-48ee-b28c-a68ddc5c3f27.png"
  >
</p>


### While comparing each facility of a state against the country average, some states like Chandigarh, have much better water facilities than India as a whole.

<p align="center">
  <img 
    width="600"
    height="400"
    src="https://user-images.githubusercontent.com/62461730/163446279-7a713289-c97d-4528-9ff9-606672ad6edb.png"
  >
</p>

### On Comparing the distribution of the facility across the country,  Bihar, Assam, Jharkhand, Meghalaya, Uttar Pradesh and Odisha heavily rely on Kerosene.

 ![image](https://user-images.githubusercontent.com/62461730/163446474-ae7e4188-480a-401f-ba6e-93715b3deca5.png)


### Filtering 

Two filtering algorithms are built to answer a question <b>What state should I live in?</b>. The principle of both the algorihtms are same, filter the states on the basis of their basic amenities and characteristics and provvide end user the results. The difference is in the results algorithm 1 gives us the count of districts in state where there are availibiity of certain amenity satisfies the user defined threshold. Algorithm 2 provides us with visualizatios of mean percentage availibility of the assets selected by the users. Algorithm 2 can be observed in the UI and Algorithm 1 can be found in the PART A folder.
 

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

