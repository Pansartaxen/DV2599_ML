import pandas
import matplotlib.pyplot as plt
import numpy as np

def convert_image(path, label, dataframe):
    img = plt.imread(path)
    row = np.array(img).flatten()
    row = np.append(row, label)
    index = dataframe.shape[0]
    dataframe.loc[len(dataframe)] = row
    return dataframe
