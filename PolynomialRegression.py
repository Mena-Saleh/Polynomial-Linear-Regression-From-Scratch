import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import combinations_with_replacement


# Custom Classes:

class PolynomialFeaturesFromScratch:

    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, X):
        # Get data shape
        rows, cols = X.shape
        # Initialize X_poly as an empty array with the shape of the data to concatenate new features on it.
        X_new = np.empty((rows, 1))

        # First we get all combinations of features using some combinatorics (combinations with replacement)
        # Then we multiply each combination of features to make a new feature and add that feature.
        # This is done in a building up manner going through all degrees up to d
        for d in range(1, self.degree + 1):
            # Get all combinations of input features up to degree d
            combinations = combinations_with_replacement(range(cols), d)
            for c in combinations:
                # Create polynomial feature as a product of input features with given combination (c has the indices of the columns to be multiplied)
                new_feature = np.prod(X[:, c], axis=1)
                # Make it 2D
                new_feature = new_feature[:, np.newaxis]
                # Concatenate the newly generated feature:
                X_new = np.concatenate((X_new, new_feature), axis=1)

        X_new = X_new[:, 1:] #Remove the zeroes column
        return X_new


class LinearRegressionFromScratch:

    def __init__(self):
        self.Theta = None

    def fit(self, x , y, l=0.0000001, epochs=1000): #Default learning rate and epochs
        n = float(len(x))  # Number of elements in x
        x = np.array(x)
        y = np.array(y)
        # Add column of ones: (Multiplied by theta 0 which always results in a constant)
        x = np.c_[np.ones((x.shape[0], 1)), x]
        # Transform Y to two dimensions by adding an extra dimension
        y = y[:, np.newaxis]
        # Store thetas initially in a zeroes array with dimensions equal no of features * 1
        theta = np.zeros((x.shape[1], 1))
        # Gradient Descent Algorithm:
        for i in range(epochs):
            # Calculate Ypred at current parameters:
            ypred = np.dot(x, theta)
            # Calculate Gradience
            djw = -(2 / n) * np.dot(x.T, y - ypred)
            # Update thetas
            theta = theta - l * djw

        self.Theta = theta

    def predict(self, x): #returns predicted Y
        # Takes 2d Data, adds one columns and applies the regression line equation to the data.
        x = np.array(x)
        x = np.c_[np.ones((x.shape[0], 1)), x]
        prediction = np.dot(x, self.Theta)
        return prediction

# Data and preprocessing:
# Reading data and dropping null if any exists:

data = pd.read_csv('fifa19.csv')
data.dropna(how='any',inplace=True)

# Feature selection based on correlation:

fifa_data = data.iloc[:, :]

corr = fifa_data.corr()

#Top 6.5% Correlation training features with the Value:
top_features = corr.index[abs(corr['Value'])>0.5]

#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

top_features = top_features.delete(-1) #Delete Value from there as we only need X column names.

# X and Y:

Y = data['Value']
X = data[top_features]


# Normalize X:

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle= True, random_state=42)

# Make polynomial features:

# Built in:
poly_features_built_in = PolynomialFeatures(degree=5)
X_train_poly_built_in = poly_features_built_in.fit_transform(X_train)
X_test_poly_built_in = poly_features_built_in.transform(X_test)

# From scratch:

poly_features_from_scratch = PolynomialFeaturesFromScratch(degree=5)
X_train_poly_from_scratch = poly_features_from_scratch.fit_transform(X_train)
X_test_poly_from_scratch = poly_features_from_scratch.fit_transform(X_test)

# Model training and prediction:

# Built in Method:
model_built_in = linear_model.LinearRegression()
model_built_in.fit(X_train_poly_built_in, Y_train)
Y_pred_built_in = model_built_in.predict(X_test_poly_built_in)


# From scratch Method:
model_from_scratch = LinearRegressionFromScratch()
model_from_scratch.fit(X_train_poly_from_scratch, Y_train, l=0.0001, epochs= 10000)
Y_pred_from_scratch = model_from_scratch.predict(X_test_poly_from_scratch)


# Evaluation:
r2_built_in = metrics.r2_score(Y_test, Y_pred_built_in)
r2_from_scratch = metrics.r2_score(Y_test, Y_pred_from_scratch)
print('R-squared of the built in method is:', r2_built_in)
print('Mean Square Error of built in is: ' , metrics.mean_squared_error(Y_test, Y_pred_built_in))
print('R-squared of from scratch is:', r2_from_scratch)
print('Mean Square Error of the from scratch method is ', metrics.mean_squared_error(Y_test, Y_pred_from_scratch))






