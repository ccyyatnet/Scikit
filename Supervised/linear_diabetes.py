import numpy as np
from sklearn import datasets,linear_model

diabetes = datasets.load_diabetes()
n = len(diabetes.data)
variable = input("Fit on variable(0-9):")

X = diabetes.data[:, variable:(variable+1)]
X_test = X[n/2:]
X_train = X[:n/2]
y = diabetes.target
y_test = y[n/2:]
y_train = y[:n/2]

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)

print "coef:"
print regr.coef_
print "intercept:"
print regr.intercept_
print "Mean Square Error:"
print np.mean((regr.predict(X_test)-y_test)**2)
print "variance score (1 is perfict linear relationship):"
print regr.score(X_test,y_test)
