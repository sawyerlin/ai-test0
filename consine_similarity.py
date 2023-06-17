import sys
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample coach dataset
coaches = [
    {
        'id': 1,
        'name': 'Coach A',
        'specialty': 'Technique, Strategy',
        'location': 'City A',
        'availability': 'Weekdays',
        'fee_rate': 50
    },
    {
        'id': 2,
        'name': 'Coach B',
        'specialty': 'Advanced Training',
        'location': 'City B',
        'availability': 'Weekends',
        'fee_rate': 70
    },
    {
        'id': 3,
        'name': 'Coach C',
        'specialty': 'Fitness, Conditioning',
        'location': 'City A',
        'availability': 'Evenings',
        'fee_rate': 60
    },
    {
        'id': 4,
        'name': 'Coach D',
        'specialty': 'Personalized Coaching',
        'location': 'City C',
        'availability': 'Weekdays',
        'fee_rate': 80
    },
    {
        'id': 5,
        'name': 'Coach E',
        'specialty': 'Mental Training, Mindset',
        'location': 'City B',
        'availability': 'Weekends',
        'fee_rate': 90
    }
]

# Sample player dataset
players = [
    {
        'id': 1,
        'name': 'Player 1',
        'wanted_specialty': 'Technique, Strategy',
        'wanted_location': 'City A',
        'wanted_avail': 'Weekdays',
        'wanted_fee': 60
    },
    {
        'id': 2,
        'name': 'Player 2',
        'wanted_specialty': 'Advanced Training',
        'wanted_location': 'City B',
        'wanted_avail': 'Weekends',
        'wanted_fee': 70
    },
    {
        'id': 3,
        'name': 'Player 3',
        'wanted_specialty': 'Fitness, Conditioning',
        'wanted_location': 'City A',
        'wanted_avail': 'Evenings',
        'wanted_fee': 50
    }
]

data1 = pd.read_json(json.dumps(coaches), orient='records')
print(data1)
print("\n")
data2 = pd.read_json(json.dumps(players), orient='records')
print(data2)
print("\n")

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

# Build the coach feature matrix
coach_descriptions = [f"{coach['specialty']} {coach['availability']} {coach['location']}" for coach in coaches]
coach_features = vectorizer.fit_transform(coach_descriptions)

# Function to get coach recommendations based on player preferences
def get_recommendations(player, top_n=3):
    # Create a query vector from player preferences
    query = vectorizer.transform([f"{player['wanted_specialty']} {player['wanted_avail']} {player['wanted_location']}"])

    # Calculate the cosine similarity scores between the query and coach features
    similarity_scores = linear_kernel(query, coach_features).flatten()

    # Get the indices of coaches sorted by similarity scores
    sorted_indices = similarity_scores.argsort()[::-1]

    # Filter coaches based on fee rate preferences
    filtered_coaches = []
    for index in sorted_indices:
        coach = coaches[index]
        if coach['fee_rate'] <= player['wanted_fee']:
            filtered_coaches.append(coach)
            if len(filtered_coaches) == top_n:
                break

    # Print the recommendations
    print(f"""Recommended coaches for player
- {player['name']}
    - Wanted Speciality: {player['wanted_specialty']}
    - Wnated Availability: {player['wanted_avail']}
    - Wanted Location: {player['wanted_location']}
    - Wanted Max Fee: {player['wanted_fee']}
Match to :""")
    for coach in filtered_coaches:
        print(f"""- {coach['name']}
    - Specialty: {coach['specialty']}
    - Availability: {coach['availability']}
    - Location: {coach['location']}
    - Fee Rate: {coach['fee_rate']}""")
    print("\n")

# Test the recommendation system
for player in players:
    get_recommendations(player, top_n=2)
