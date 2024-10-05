import matplotlib.pyplot as plt
import pandas as pd

# Load the iris dataset
iris_dataset = pd.read_csv("iris.csv")

# Group the data by 'variety'
iris_dataset_group = iris_dataset.groupby("variety")[["sepal.length","sepal.width","petal.length","petal.width"]]

# Plotting box plots for each feature and comparing them across species
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Creating boxplots
iris_dataset.boxplot(column="sepal.length", by="variety", ax=axes[0, 0])
axes[0, 0].set_title("Sepal Length")
axes[0, 0].set_xlabel("Species")
axes[0, 0].set_ylabel("Sepal Length")

iris_dataset.boxplot(column="sepal.width", by="variety", ax=axes[0, 1])
axes[0, 1].set_title("Sepal Width")
axes[0, 1].set_xlabel("Species")
axes[0, 1].set_ylabel("Sepal Width")

iris_dataset.boxplot(column="petal.length", by="variety", ax=axes[1, 0])
axes[1, 0].set_title("Petal Length")
axes[1, 0].set_xlabel("Species")
axes[1, 0].set_ylabel("Petal Length")

iris_dataset.boxplot(column="petal.width", by="variety", ax=axes[1, 1])
axes[1, 1].set_title("Petal Width")
axes[1, 1].set_xlabel("Species")
axes[1, 1].set_ylabel("Petal Width")

# Adjust the layout
plt.tight_layout()
plt.suptitle("Distribution of Features by Species", y=1.02)
plt.show()

