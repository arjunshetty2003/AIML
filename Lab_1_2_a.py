import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
df = pd.read_csv("iris.csv")

# Check the contents of the DataFrame
print(df.head())
print("Columns in the DataFrame:", df.columns)
print("Missing values in each column:", df.isnull().sum())
print("Species counts:", df['species'].value_counts())

# Ensure no leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# 1. Scatter plot of sepal length vs sepal width colored by species
plt.figure(figsize=(10, 6))
species_colors = {'setosa': 'blue', 'versicolor': 'orange', 'virginica': 'green'}

# Plot each species with a different color
for species, color in species_colors.items():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal.length'], subset['sepal.width'], 
                color=color, label=species, s=100)

plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.grid(True)
plt.show()

# 2. Histogram of petal lengths
plt.figure(figsize=(10, 6))
plt.hist(df['petal.length'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Petal Lengths')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# 3. Box plots comparing distributions of each feature for different species
features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

plt.figure(figsize=(15, 10))

for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)
    # Create a box plot for each feature
    df.boxplot(column=feature, by='species', grid=False)
    plt.title(f'Box Plot of {feature.replace(".", " ").title()} by Species')
    plt.suptitle('')  # Suppress the default title
    plt.xlabel('Species')
    plt.ylabel(feature.replace(".", " ").title())

plt.tight_layout()
plt.show()