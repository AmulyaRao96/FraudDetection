# FraudDetection
This project consists of having applied various Machine Learning Algorithms to build a model that predicts the fraudulent money transactions based on a month-long mobile money transaction in an African Country.
The dataset for building the model has been obtained from Kaggle. The dataset has been cleaned, after which the Exploratory Data Analysis has been carried out to get visual insight into the dataset and the different features in it. Various Machine Learning Algorithms have been implemented to understand the model performance. This entire model building process is carried out in Jupyter Notebook (Anaconda 3).
After carefully going through the dataset, it becomes evident that it is a classification type problem with target variable ‘isFraud’ taking values of either 0 or 1. The different Machine Learning Algorithms used for modelling are Logistic Regression, Decision Tree Classifier, Random Forest Classifier and K-Nearest Neighbors.
The model built using different algorithms is tested for its accuracy and is compared to obtain the best model that can correctly classify the fraudulent transactions.

Financial Fraud Detection aims to predict a fraudulent money transaction which is based on sample of real transactions extracted from one month of financial logs from a mobile money service implementation in an African country. This fraud detection system has ‘isFraud’ as the target variable which is detected based on a number of features present in the dataset like the type of transaction made, the amount being transferred, the balance before and after transaction of the person transferring the money and the person who receives the money

About the Dataset:

The dataset consists of one month of financial log of mobile money service deployed in an African country. The dataset has a size of about 493MB.It has 6362620 rows and 11 columns.

Software and Libraries used
The dataset is downloaded from Kaggle. The Jupyter notebook is used with several Libraries
They are as follow:
•	Numpy 
•	Pandas 
•	Matplotlib
•	Seaborn 
•	Metrics
•	Mean Squared Error
•	Scikitlearn Tree
•	DecisionTreeClassifier
•	train_test_split 
•	accuracy_score 
•	sklearn. ensemble 
•	RandomForestClassifier
•	sklearn.preprocessing
•	LabelEncoder
•	OneHotEncoder
•	sklearn. pipeline
•	GridSearchCV,
•	RandomizedSearchCV
•	sklearn. utils

Conclusion:
The accuracy score of Decision Tree Classifier is best with the accuracy of 91% in predicting the target variable ‘isFraud’.
•	From the analysis carried out we see that fraudulent transactions occur only due to Cash_out and Transfer type of transactions.
•	Fraudulent transactions occur between Customer-to-Customer transactions.
