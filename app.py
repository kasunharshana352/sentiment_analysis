from flask import Flask, request, jsonify, render_template
import joblib  # Import joblib to load the model and vectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the request
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform the input text using the loaded vectorizer
    text_tfidf = vectorizer.transform([text])

    # Make a prediction using the loaded model
    prediction = model.predict(text_tfidf)

    # Return the prediction as JSON
    return jsonify({"sentiment": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
