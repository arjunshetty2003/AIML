import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

wine_dataset = pd.read_csv("wine.csv")

# Printing first few rows of dataset
print("First five rows")
print(wine_dataset.head())

correlation_matrix = wine_dataset.corr()

# Display the correlation matirx
print("\nCorrelation Matrix")
print(correlation_matrix)

# Correlation with wine quality
wine_correlations = correlation_matrix["Wine"].sort_values(ascending=False)
print("Wine correlations")
print(wine_correlations)

# Creating heatmap using Seaborn
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix of Winde Dataset")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()