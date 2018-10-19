import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# Read data
dataframe = pd.read_fwf('brain_body.txt')
print(dataframe)
