To implement pythoncode we used anaconda jupyter noteboak.

Libraris Used: 
Pandas, Numpy, MatplotLib, ScikitLearn. 
train and test dataset are used. 
train dataset will be used to train model. It will have all labels assigned to it.  its used to develop model and test its accuracy. 
test dataset wont have predefined labels. Using model we built we will find assigned class or labels for test dataset.
Dataset values graphs show it hwas skewed graph and has a lot odf outliers. So normaization was done using log values. 
Missing values were replaced with finding mode of the exisiting values for categorical data, mean and log values for numerical data.
Independent variables, Dependent variable(status with Yes/No) were found. This is how dataset was divided with help of index values.
Dataset was split into Train and Test dataset using sklearn's train_test_split. 80 % training and 20 % testing data (ratio 80:20)
random state was set to 0 so that result and accuracy wont change in every cycle of execution of models.
 categorical data was converted to numeric format 0,1 using label encoder so that machine can understand.
Later dataset was scaled  for machine learning algorithm usage for better analysis and prediction.
This above is preprocessing part on dataset.
Lateral part of the project is application of various machine learning  algorithms for prediction of values of test dataset 
      Decision Tree, Random Forest, Naive Bayes, KNeighbors algorithms.
then accuracy of these predictions against actual values algorithms was validated.
Naive Baysed algorithm showed best accuracy.
