# MiniProject2
* What the project does:
    * This project is for practicing preprocessing, selecting and training proper model, and evaluating/analyzing these model with the real-world data/problem. In this project, we found the proper insuarnce charges for each customer based on the model we trained with old data given by the health insurance company to solve the problem of current charges about underpricing high-risk customers and overpricing low-risk customers.
    * We used various supervised model (simple linear regression, multiple linear regression, polynominal regression, Ridge regression, Lasso regression, Support Vector Regression, Decision Tree regression) to find the best tool(model) to predict medical charges. We compared the errors from each models and chose the best one.
    * We also used Logistic regression model to do the classification for customers for grouping 'expensive' customer or 'not expensive' customer.
* How to download the dataset from Kaggle
    * We went to the Kaggle website where insuance data is given using the provided link and downloaded it manually to our local machine and uploaded this data to github. We used the file path for the insurance data file we uploaded in github to load the data (which is just ./insurance.csv for us since this file is in the same level with jupyter notebook file for miniproject2).
* How to run the notebook
    * Assume that all necessary python libraries are installed in conda environment, we first activate the conda environment. 
    * THen, select the proper python enviornment for Jupyter file (python version of 3.12.10, which should be installed in conda). 
    * To run the entire codes in Jupyter notebook, we can click 'Run All' button on the top of the jupyter notebook file. If we want to run each cell seperately, we can press 'Execute Cell' button on left side of the cell. 
* Brief summary of the findings
    * Smoking affects a lot for predicting charge since it has the highest correlation with charge in inputs.
    * Age and bmi affect less for predicting charge than smoking, but affects meaningfully for predicting charge than children, sex, and region.
    * Children, sex, and region have low correlation with charge, so they hardly affects on predicting charge.
    * Since the all influential inputs and output(charge) have linear relationship, linear model has a higher rate of accuracy or has a less error for predicting charges than using non-linear model like polynominal with 2 degree/rtf, etc, but it has risk of overfitting.
    * More findings are in technical report documentation.