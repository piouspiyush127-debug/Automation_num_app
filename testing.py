import unittest
import pandas as pd
import numpy as np
from io import StringIO

# Assume the functions have been imported from the module
# from your_module import load_data, preprocess_data, test_multiple_algorithms, generate_report

class TestAIEmployee(unittest.TestCase):

    def setUp(self):
        # Set up a small dataset for testing
        data = """Country,Gold,Silver,Bronze,Total
        USA,46,37,38,121
        Great Britain,27,23,17,67
        China,26,18,26,70
        Russia,19,18,19,56
        Germany,17,10,15,42
        """
        self.df = pd.read_csv(StringIO(data))

    def test_load_data_csv(self):
        # Mock CSV file load
        csv_data = StringIO("""A,B,C\n1,2,3\n4,5,6\n""")
        df = load_data(csv_data)
        self.assertEqual(df.shape, (2, 3))  # 2 rows, 3 columns

    def test_preprocess_data(self):
        preprocessed_data, categorical_cols, numeric_cols = preprocess_data(self.df)
        
        # Check if duplicates are removed
        self.assertEqual(preprocessed_data.shape[0], 5)  # 5 unique rows

        # Check if categorical and numeric columns are identified correctly
        self.assertEqual(list(categorical_cols), ['Country'])
        self.assertEqual(list(numeric_cols), ['Gold', 'Silver', 'Bronze', 'Total'])

        # Check if null values are dropped
        self.assertEqual(preprocessed_data.isnull().sum().sum(), 0)

    def test_test_multiple_algorithms(self):
        # We will check if the function runs without error
        try:
            test_multiple_algorithms(self.df, 'Gold')
        except Exception as e:
            self.fail(f"test_multiple_algorithms raised an exception: {e}")

    def test_generate_report(self):
        # Test if the report generation functions can run without exceptions
        try:
            generate_report(self.df, 'Gold')
        except Exception as e:
            self.fail(f"generate_report raised an exception: {e}")

    def test_process_query(self):
        # Test process_query for known queries
        process_query("Can you analyze the data?")  # Should pass without error
        process_query("What's the report?")  # Should pass without error
        process_query("Exit the program.")  # Should pass without error

        # Test an unknown query
        with self.assertLogs(level='INFO') as log:
            process_query("Unknown command")
            self.assertIn("Sorry, I couldn't understand your query.", log.output[-1])

if __name__ == "__main__":
    unittest.main()