import matplotlib.pyplot as plt
import pandas as pd

# Reading data
df = pd.read_csv("iris.csv")

# Grouping Data
df1 = df.groupby("variety")[["sepal.length","sepal.width"]]
sepal_length_array = {}
sepal_width_array = {}

for name,group in df1 :
    sepal_length_array[name] = group["sepal.length"].values
    sepal_width_array[name] = group["sepal.width"].values

# Generating arrays for datasets
x_axis_setosa = sepal_length_array["Setosa"]
y_axis_setosa = sepal_width_array["Setosa"]

x_axis_versicolor = sepal_length_array["Versicolor"]
y_axis_versicolor = sepal_width_array["Versicolor"]

x_axis_virginica = sepal_length_array["Virginica"]
y_axis_virginica = sepal_width_array["Virginica"]

# Plotting graph
plt.scatter(x_axis_setosa, y_axis_setosa, color="red", label="Setosa")
plt.scatter(x_axis_versicolor, y_axis_versicolor, color="green", label="Versicolor")
plt.scatter(x_axis_virginica, y_axis_virginica, color="blue", label="Virginica")
plt.xticks()
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Variation between Sepal Length and Width in three varities of Iris")
plt.legend()
plt.grid(True)
plt.show()