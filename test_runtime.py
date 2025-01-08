import unittest
import pandas as pd
import numpy as np
from main_runtime import TheAlgorithm, X_train, y_train
import sys

# Ausgabe in die Datei umleiten
sys.stdout = open('/content/ausgabe2.txt', 'w')

class TestAlgorithm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load test data from CSV files
        cls.X_test = pd.read_csv('test_data.csv')
        cls.y_test = pd.read_csv('test_labels.csv')

        # Initialize the algorithm class with loaded test data
        cls.algo = TheAlgorithm(X_train, y_train, cls.X_test, cls.y_test)
        
        # Load saved reference accuracy and confusion matrix
        with open('reference_accuracy.txt', 'r') as f:
            cls.reference_accuracy = float(f.read())
        
        cls.reference_confusion_matrix = pd.read_csv('reference_confusion_matrix.csv').values

        # Run the fit() function once and store the representative runtime
        cls.representative_runtime = cls.algo.fit()

    def test_predict_accuracy(self):
        """Test if the predict function returns correct accuracy and confusion matrix."""
        test_accuracy = self.algo.predict()
        self.assertAlmostEqual(
            test_accuracy, self.reference_accuracy, delta=0.5,
            msg="The accuracy of the predict() function deviates from the reference."
        )
        self.assertTrue(
            np.array_equal(self.algo.test_confusion_matrix, self.reference_confusion_matrix),
            "The confusion matrix of the predict() function deviates from the reference."
        )

    def test_fit_runtime(self):
        """Test if the fit function runs within 120% of the representative runtime."""
        import time
        start_time = time.time()
        self.algo.fit()
        elapsed_time = time.time() - start_time
        max_allowed_time = self.representative_runtime * 1.2
        self.assertLessEqual(
            elapsed_time, max_allowed_time,
            "The runtime of the fit() function exceeds the threshold."
        )

# Run the unit tests
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# Ausgabe zurück auf die Konsole umleiten
sys.stdout.close()
sys.stdout = sys.__stdout__

