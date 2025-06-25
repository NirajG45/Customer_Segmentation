from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Load and preprocess the data
    df = pd.read_csv('Mall_Customers.csv')
    data = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(data)

    # Save cluster plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
    plt.title('Customer Segments')
    plt.savefig('static/cluster_plot.png')
    plt.close()

    # Save CSV output
    segmented_data = df.copy()
    segmented_data.to_csv('static/segmented_customers.csv', index=False)

    # 🧼 Clean data for HTML table rendering
    segmented_data = segmented_data.applymap(
        lambda x: str(x).replace('\n', ' ').strip() if isinstance(x, str) else x
    )

    # Data for Chart.js
    cluster_counts = segmented_data['Cluster'].value_counts().to_dict()
    income_by_cluster = segmented_data.groupby('Cluster')['Annual Income (k$)'].mean().round(2).to_dict()
    scatter_data = segmented_data[['Age', 'Spending Score (1-100)', 'Cluster']].values.tolist()

    return render_template('index.html',
        tables=[segmented_data.to_html(classes='data', escape=False)],
        title='Customer Segmentation',
        image='static/cluster_plot.png',
        cluster_counts=cluster_counts,
        income_by_cluster=income_by_cluster,
        scatter_data=scatter_data
    )

@app.route('/download')
def download():
    return send_file('static/segmented_customers.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
