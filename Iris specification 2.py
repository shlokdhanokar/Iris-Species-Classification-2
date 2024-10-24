# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data  # features
y = iris.target  # target labels (species)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # You can choose a different value for n_neighbors
knn.fit(X_train, y_train)

# Step 4: Evaluate the classifier
y_pred = knn.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Step 5: Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Visualizing the results using pairplot
iris_df = pd.DataFrame(data=np.c_[iris.data, iris.target], columns=iris.feature_names + ['species'])
iris_df['species'] = iris_df['species'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})

sns.pairplot(iris_df, hue='species', palette='husl')
plt.suptitle('Pairplot of Iris Species', y=1.02)
plt.show()
