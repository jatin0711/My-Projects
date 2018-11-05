import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('train_CSRqzyo.csv')

X_train=dataset.drop("Item_Outlet_Sales")



from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
""" 
plotting of training set
"""
plt.scatter(X_train,y_train)
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("salary vs year of experience")
plt.xlabel("salary")
plt.ylabel("year of experience")
plt.show()

""" 
plotting of test set
"""
plt.scatter(X_test,y_test)
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("salary vs year of experience")
plt.xlabel("salary")
plt.ylabel("year of experience")
plt.show()
