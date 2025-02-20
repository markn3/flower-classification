import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/Iris.csv")

print(df)
print(type(df))
print(df['Species'].unique())

# # map target numbers to species
print(df)

# Pairplot to visualize feature distribution by species
sns.pairplot(df, hue="Species")
plt.show()