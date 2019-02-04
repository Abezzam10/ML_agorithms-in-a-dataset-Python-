import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['Trainee_l', 'Trainee_w', 'cross_l', 'cross_w', 'class']
dataset = pandas.read_csv(url, names=names)

print(dataset.shape)

print(dataset.head(20))

print(dataset.describe())


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) # Plot univariate Plots
plt.show()


scatter_matrix(dataset) # scatter plot matrix
plt.show()


array = dataset.values  # Training and cross validation set 
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


'''Applying logistic regression, Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), Classification and Regression Trees (CART), Gaussian Naive Bayes (NB), Support Vector Machines (SVM)!'''

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))  # Linear regression

models.append(('LDA', LinearDiscriminantAnalysis())) # linear discriminant analysis

models.append(('KNN', KNeighborsClassifier()))# K-Nearest Neighbors

models.append(('CART', DecisionTreeClassifier()))# Classification and regression

models.append(('NB', GaussianNB())) # Gaussian Naive Bayes

models.append(('SVM', SVC(gamma='auto'))) # Support vector machine

# storing results in a basic list

results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# accuracy for Knn__
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


