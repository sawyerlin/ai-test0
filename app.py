from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    return "ai test"

@app.route('/recommend', methods=["POST"])
def recommend():
    data = request.get_json()
    coaches = data.get('coaches')
    player = data.get('player')

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

    # Build the coach feature matrix
    coach_descriptions = [f"{coach['specialty']} {coach['availability']} {coach['location']}" for coach in coaches]
    coach_features = vectorizer.fit_transform(coach_descriptions)

    query = vectorizer.transform([f"{player['wanted_specialty']} {player['wanted_avail']} {player['wanted_location']}"])
    similarity_scores = linear_kernel(query, coach_features).flatten()
    sorted_indices = similarity_scores.argsort()[::-1]
    filtered_coaches = []
    for index in sorted_indices:
        coach = coaches[index]
        if coach['fee_rate'] <= player['wanted_fee']:
            filtered_coaches.append(coach)
            if len(filtered_coaches) == 2:
                break
    return jsonify(filtered_coaches)


if __name__ == '__main__':
    app.run()
