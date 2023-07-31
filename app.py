from flask import Flask, render_template, jsonify, request
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Tokenize and preprocess the text
def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()

    stop_words = set(stopwords.words('english'))

    words = [word for word in words if word not in stop_words]

    preprocessed_text = " ".join(words)

    return preprocessed_text

def get_word_frequencies(text):

    words = text.lower().split()

    word_freq = Counter(words)

    return word_freq


@app.route('/')
def index():
    global text 
    text = [
    "The sun sets behind the distant mountains.",
    "The playful puppy chased its tail in circles.",
    "She smiled warmly at the elderly couple sitting on the park bench.",
    "The aroma of freshly brewed coffee filled the air.",
    "With a flick of his wrist, he caught the flying baseball.",
    "The sound of waves crashing on the shore was calming.",
    "The city skyline looked breathtaking in the evening light.",
    "The students eagerly listened to the fascinating lecture.",
    "The chef skillfully prepared a delicious gourmet meal.",
    "The ancient ruins held many secrets of the past.",
    "The chirping of birds greeted the morning with joy.",
    "The bookshelf was filled with a vast collection of novels.",
    "The leaves rustled in the gentle breeze of autumn.",
    "The laughter of children echoed through the playground.",
    "The majestic waterfall cascaded down the rocky cliff.",
    "The musician played a soulful melody on the violin.",
    "The rainbow stretched across the sky after the rain.",
    "The adventurous explorers ventured deep into the jungle.",
    "The artist painted a masterpiece on the blank canvas.",
    "The starlit sky shone brightly on a clear night.",
    "The astronaut floated gracefully in the zero-gravity.",
    "The delicate flower bloomed in the early spring.",
    "The detective cleverly solved the mysterious case.",
    "The crowd cheered enthusiastically at the concert.",
    "The fragrance of flowers filled the garden with sweetness.",
    "The hiker trekked through the rugged mountain terrain.",
    "The magic show left the audience in awe and wonder.",
    "The sailboat glided smoothly across the serene lake.",
    "The newborn baby slept peacefully in the crib.",
    "The wise old owl observed everything from the treetop.",
]


    preprocessed_text = [preprocess_text(t) for t in text]


    word_frequencies = get_word_frequencies(' '.join(preprocessed_text))

    vectorizer = TfidfVectorizer()

    tfidf_scores = vectorizer.fit_transform(preprocessed_text)

    feature_names = vectorizer.get_feature_names_out()
    word_indices = {word: idx for idx, word in enumerate(feature_names)}

    combined_importance_dict = {}
    for word, freq in word_frequencies.items():
        idx = word_indices[word]
        tfidf_score = tfidf_scores[:, idx].sum()  
        combined_importance = freq + tfidf_score  
        combined_importance_dict[word] = combined_importance

    sorted_items = sorted(combined_importance_dict.items(), key=lambda item: item[1], reverse=True)

    top_words = sorted_items[:20]

    word_data = [{'word': word, 'frequency': int(round(freq))} for word, freq in top_words]

    return render_template('index.html', word_data=word_data)


@app.route('/process_word', methods=['POST'])
def process_word():
    data = request.get_json()
    word = data.get('word')

    for i in text:
        if word in i:
            return jsonify({'message': f'You clicked on "{i}".'})

if __name__ == '__main__':
    app.run(debug=True)
