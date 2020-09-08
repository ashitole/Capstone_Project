#!/usr/bin/env python
"""
model tests
"""


import unittest

## import model specific functions and variables
from model import *
import sys, os
sys.path.insert(1, os.path.join('..', os.getcwd()))
data_dir=r"C:\Users\AshwiniShitole\Desktop\Ashwini\Personal\Data Science\AI Academy\AI Enterprise Workflow Certification\AI in Production\Capstone_Project\case-study-soln\data\cs-train"
saved_model= r"C:\Users\AshwiniShitole\Desktop\Ashwini\Personal\Data Science\AI Academy\AI Enterprise Workflow Certification\AI in Production\Capstone_Project\case-study-soln\models\sl-united_kingdom-0_1.joblib"
class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(data_dir,test=True)
        self.assertTrue(os.path.exists(saved_model))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## load model first
        more_data,models = model_load()
        model = list(models.values())[0]
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        #more_data,models = model_load()
    
        ## ensure that a list can be passed
        query = {'country': ['united_states','singapore','united_states'],
                 'age': [24,42,20],
                 'subscriber_type': ['aavail_basic','aavail_premium','aavail_basic'],
                 'num_streams': [8,17,14]
        }
        country='all'
        year='2018'
        month='01'
        day='05'
        result = model_predict(country,year,month,day,test=True)
        y_pred = result['y_pred']
        self.assertTrue(y_pred is not None)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
