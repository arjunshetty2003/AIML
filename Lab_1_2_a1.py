import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris_dataset = pd.read_csv("iris.csv")
# Sturges' rule for 
k = int(np.ceil(np.log2(len(iris_dataset))+1))

# Plotting graph
plt.hist(iris_dataset["petal.length"], bins=k, color="blue")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram data for petal lengths")
plt.grid(True)
plt.show()