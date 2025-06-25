from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/cluster', methods=['POST'])
def cluster():
    file = request.files['file']
    if not file:
        return "No file uploaded."

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)

    if 'CustomerID' in df.columns:
        df.drop('CustomerID', axis=1, inplace=True)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    elbow_path = os.path.join(PLOT_FOLDER, 'elbow.png')
    plt.savefig(elbow_path)

    # Final Clustering (k=5)
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # 2D Plot using Annual Income and Spending Score (if exists)
    if 'Annual Income (k$)' in df.columns and 'Spending Score (1-100)' in df.columns:
        plt.figure()
        plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='tab10')
        plt.xlabel('Annual Income')
        plt.ylabel('Spending Score')
        plt.title('Customer Segments')
        cluster_path = os.path.join(PLOT_FOLDER, 'clusters.png')
        plt.savefig(cluster_path)
    else:
        cluster_path = None

    preview_table = df.head().to_html(classes='table table-striped', index=False)

    return render_template("result.html", elbow='static/elbow.png', cluster='static/clusters.png', table=preview_table)

if __name__ == '__main__':
    app.run(debug=True)

