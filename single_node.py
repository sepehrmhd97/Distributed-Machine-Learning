from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
covtype = fetch_covtype()
X_train, X_test, y_train, y_test = train_test_split(covtype.data, covtype.target, test_size=0.2)

# Create a Random Forest Classifier with default parameters and train it
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Explore the parameters used in the Random Forest implementation
print(rfc.get_params())

# Evaluate the model on the test set
accuracy = rfc.score(X_test, y_test)
print(f"Accuracy on test set: {accuracy:.3f}")
