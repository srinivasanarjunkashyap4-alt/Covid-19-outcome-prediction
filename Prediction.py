import os
import time
import joblib
import warnings
import pandas as pd
warnings.filterwarnings(action='ignore')
from sklearn.linear_model import LogisticRegression
def predict_Covid(csv_file_path, model_file_path='logreg_model.joblib'):
    # Load the lr model
    loaded_lr = joblib.load(model_file_path)
    
    # Load the test data from the CSV file
    test_data = pd.read_csv(csv_file_path, header=None)  # Assuming no header in the CSV file
    
   
   
    predictions = loaded_lr.predict(test_data)
    
    return predictions


def eeg_model(myfile):
    data = myfile.read()
    eeg_values = [float(val) for val in data.split("\n") if val.strip()]
    clf = joblib.load('logreg_model.joblib')  # Load the trained KNN model for EEG
    new_pred = clf.predict([eeg_values])
    
    grip_mapping = {
        0: 'No Covid',
        1: 'Covid',
        }
    
    result = grip_mapping.get(new_pred[0], 'Unknown')
    
    with open("output.txt", "w") as f:
        f.write(result)
    
    print("EEG Result:", result)
    return result


