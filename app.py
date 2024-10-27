from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Load questions from CSV
def load_questions():
    return pd.read_csv('questions.csv')

# Initialize models
model = None
dl_model = None

# Load or initialize machine learning model
def load_logistic_model():
    global model
    try:
        model = joblib.load('performance_predictor.pkl')
        print("Logistic Regression model loaded successfully.")
    except FileNotFoundError:
        print("Logistic Regression model not found; it will be trained.")

# Load or initialize deep learning model
def load_deep_learning_model():
    global dl_model
    try:
        dl_model = keras.models.load_model('performance_predictor_dl.h5')
        print("Deep Learning model loaded successfully.")
    except (FileNotFoundError, OSError):
        print("Deep Learning model not found; it will be trained.")

# Train models on user responses
def train_model():
    global model, dl_model
    data = pd.read_csv('user_responses.csv')

    if data.empty or len(data) < 10:
        print("Not enough data to train the models.")
        return

    if 'score' in data.columns and 'subject' in data.columns:
        # Convert categorical data to numeric
        data['subject'] = data['subject'].astype('category').cat.codes  # Convert subject to numeric
        data['question_id'] = data['question_id'].astype('category').cat.codes  # Convert question_id to numeric
        X = data[['subject', 'question_id']]
        y = (data['score'] > 4).astype(int)  # Binary encoding for the target

        if len(X) == 0:
            print("No data available after encoding.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Logistic Regression model
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'performance_predictor.pkl')
        model = clf
        print("Logistic Regression model trained successfully.")

        # Train Deep Learning model
        input_shape = X.shape[1]

        dl_model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        dl_model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

        dl_model.save('performance_predictor_dl.h5')
        print("Deep Learning model trained successfully.")

# Log assessment attempt for time series analysis
def log_assessment_attempt(subject):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_data = pd.DataFrame([[timestamp, subject]], columns=['timestamp', 'subject'])
    log_data.to_csv('assessment_log.csv', mode='a', header=not pd.io.common.file_exists('assessment_log.csv'), index=False)
    print("Assessment attempt logged successfully.")

# Time-series analysis of subject popularity
def analyze_subject_popularity():
    try:
        data = pd.read_csv('assessment_log.csv', parse_dates=['timestamp'])
        data['date'] = data['timestamp'].dt.date  # Extract the date for daily trends
        subject_counts = data.groupby(['date', 'subject']).size().unstack(fill_value=0)

        # Plot the time series
        subject_counts.plot(kind='line', marker='o', figsize=(12, 6))
        plt.title('Daily Popularity of Subjects Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Assessments')
        plt.legend(title='Subjects')
        plt.grid()
        plt.show()

    except FileNotFoundError:
        print("No assessment log found. Please take an assessment to generate data.")

@app.route('/')
def index():
    subjects = ['Mathematics', 'Life Science', 'IT', 'Engineering', 'History']
    return render_template('index.html', subjects=subjects)

@app.route('/assessment', methods=['GET', 'POST'])
def assessment():
    if request.method == 'POST':
        subject = request.form['subject']
        log_assessment_attempt(subject)  # Log the assessment attempt for time-series analysis
        df = load_questions()
        questions = df[df['assessment'] == subject].sample(n=8).to_dict(orient='records')
        return render_template('assessment.html', subject=subject, questions=questions)
    return render_template('assessment.html', subject=None, questions=None)

@app.route('/results', methods=['POST', 'GET'])
def results():
    if request.method == 'POST':
        user_answers = request.form.to_dict()
        subject = user_answers.pop('subject')
        df = load_questions()
        correct_answers = df[df['assessment'] == subject].set_index('question')['answer'].to_dict()

        score = 0
        corrections = {}
        resources = []

        for question, user_answer in user_answers.items():
            user_answer = user_answer.strip().lower()
            correct_answer = correct_answers.get(question).lower()

            if user_answer == correct_answer:
                score += 1
            else:
                corrections[question] = (user_answer, correct_answer)
                resources.append(df[df['question'] == question]['resources'].values[0])

        performance_category = categorize_performance(score, len(correct_answers))
        return render_template('results.html', score=score, total=len(correct_answers),
                               corrections=corrections, resources=resources, 
                               performance_category=performance_category)
    
    return render_template('results.html')

@app.route('/analyze_popularity')
def analyze_popularity():
    analyze_subject_popularity()
    return redirect(url_for('index'))

def categorize_performance(score, total):
    percentage = (score / total) * 100
    if percentage >= 80:
        return "Excellent"
    elif percentage >= 60:
        return "Good"
    elif percentage >= 40:
        return "Average"
    else:
        return "Needs Improvement"

if __name__ == '__main__':
    load_logistic_model()
    load_deep_learning_model()
    if model is None or dl_model is None:
        train_model()
    app.run(debug=True)
