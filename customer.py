import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Step 1: Load Data
df = pd.read_csv("Mall_Customers.csv")

# Step 2: Preprocess
df = df.drop('CustomerID', axis=1)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Step 3: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 4: Elbow Method to Find Optimal k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Ensure plots folder exists
os.makedirs("plots", exist_ok=True)
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("plots/elbow_plot.png")
plt.show()

# Step 5: Apply K-Means
k = 5  # Based on elbow result
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 6: Save Output
df.to_csv("Clustered_Customers.csv", index=False)

# Step 7: Visualize Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette='tab10')
plt.title("Customer Segments")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.savefig("plots/cluster_plot.png")
plt.show()
