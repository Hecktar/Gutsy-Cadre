from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Sample student data
data = {
    'student_id': [1, 2, 3, 4, 5],
    'learning_style': ['visual', 'auditory', 'kinesthetic', 'auditory', 'visual'],
    'performance': [75, 60, 80, 55, 90],  # Scores in percentages
    'engagement': [0.9, 0.7, 0.85, 0.5, 0.95]  # Engagement levels from 0 to 1
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing: Normalize engagement (engagement in 0-1 to 0-100 scale)
df['engagement'] = df['engagement'] * 100

# Features: Using performance and engagement to cluster students
X = df[['performance', 'engagement']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Assign personalized content based on cluster
def recommend_content(cluster):
    if cluster == 0:
        return "Interactive videos and quizzes"
    elif cluster == 1:
        return "Reading material and audiobooks"
    else:
        return "Hands-on activities and projects"

# Function to get recommendation based on input data
def get_recommendation(performance, engagement):
    # Normalize and predict the cluster
    input_data = scaler.transform([[performance, engagement]])
    cluster = kmeans.predict(input_data)[0]
    return recommend_content(cluster)

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API route to get recommendation
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    performance = data['performance']
    engagement = data['engagement'] * 100  # Convert engagement to percentage
    
    # Get personalized recommendation
    recommendation = get_recommendation(performance, engagement)
    
    return jsonify({'recommendation': recommendation})

if __name__ == "__main__":
    app.run(debug=True)
