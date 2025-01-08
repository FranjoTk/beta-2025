import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from decorators import my_logger, my_timer
import sys

# Ausgabe in die Datei umleiten
sys.stdout = open('/content/ausgabe2.txt', 'w')

# Load dataset
data = pd.read_csv('Advertising.csv')

# Select relevant features and target
X = data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = data['Clicked on Ad']

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Algorithm class
class TheAlgorithm:
    
    @my_logger
    @my_timer
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.scaler = MinMaxScaler()
        self.classifier = LogisticRegression(random_state=42)
    
    @my_logger
    @my_timer
    def fit(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.classifier.fit(self.X_train, self.y_train)
        self.train_accuracy = self.classifier.score(self.X_train, self.y_train) * 100
        self.train_confusion_matrix = confusion_matrix(self.y_train, self.classifier.predict(self.X_train))
        return self.train_accuracy
    
    @my_logger
    @my_timer
    def predict(self):
        self.test_accuracy = self.classifier.score(self.X_test, self.y_test) * 100
        self.test_confusion_matrix = confusion_matrix(self.y_test, self.classifier.predict(self.X_test))
        self.report = classification_report(self.y_test, self.classifier.predict(self.X_test))
        print("Classification report:\n", self.report)
        return self.test_accuracy

# Instantiate the class and run fit and predict
if __name__ == '__main__':
    ta = TheAlgorithm(X_train, y_train, X_test, y_test)
    train_accuracy = ta.fit()
    print("\nTrain Accuracy:", train_accuracy)
    print("Train Confusion Matrix:\n", ta.train_confusion_matrix)

    test_accuracy = ta.predict()
    print("\nTest Accuracy:", test_accuracy)
    print("Test Confusion Matrix:\n", ta.test_confusion_matrix)

    # Export test data to CSV files
    X_test.to_csv('test_data.csv', index=False)
    y_test.to_csv('test_labels.csv', index=False)

    # Save reference accuracy and confusion matrix
    reference_accuracy = test_accuracy
    reference_confusion_matrix = pd.DataFrame(ta.test_confusion_matrix)

    reference_accuracy_file = 'reference_accuracy.txt'
    reference_confusion_matrix_file = 'reference_confusion_matrix.csv'

    # Write reference accuracy to a text file
    with open(reference_accuracy_file, 'w') as f:
        f.write(str(reference_accuracy))

    # Write reference confusion matrix to a CSV file
    reference_confusion_matrix.to_csv(reference_confusion_matrix_file, index=False)

    print(f"\nTest data and reference values have been exported:\n"
          f"- Test data: test_data.csv, test_labels.csv\n"
          f"- Reference accuracy: {reference_accuracy_file}\n"
          f"- Reference confusion matrix: {reference_confusion_matrix_file}")

# Ausgabe zurück auf die Konsole umleiten
sys.stdout.close()
sys.stdout = sys.__stdout__

