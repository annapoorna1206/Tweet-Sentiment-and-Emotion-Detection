from textblob import TextBlob
from flask import Flask, render_template, request
import re
import nltk
from nltk.corpus import stopwords
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize app
app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load tokenizer and model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

emotion_model = load_model('tweet_emotion_model.h5')
MAX_LENGTH = 50  # Use the same maxlen used during training

# Emotion label map (adjust this to your dataset)
label_map = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}

# Preprocessing functions
stp = stopwords.words('english')

def remove_usernames_links(tweet):
    s2 = re.sub('http://\S+|https://\S+', '', tweet)
    s1 = re.sub(r"#[a-zA-Z0-9\\n@_\s]+", "", s2)
    return s1   

def remove_emoji(txt):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
        "]+", re.UNICODE)
    return re.sub(emoj, '', txt)

def TweetCleaning(tweet):
    link_removal = remove_usernames_links(tweet)
    emoji_removal = remove_emoji(link_removal)
    after_stopword_removal = ' '.join(word for word in emoji_removal.split() if word.lower() not in stp)
    return after_stopword_removal

# Sentiment Analysis
def AnalysSentiment(tweet):
    sentence = TweetCleaning(tweet)
    blob = TextBlob(sentence)
    sentiment = blob.sentiment
    return sentiment

def segmentation(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Emotion Detection
def predict_emotion(tweet):
    # cleaned = TweetCleaning(tweet)
    seq = tokenizer.texts_to_sequences([tweet])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')
    pred = emotion_model.predict(padded)
    pred_class = pred.argmax(axis=-1)[0]
    return label_map[pred_class]

# Flask Routes
@app.route('/', methods=['GET','POST'])
def home():
    sentiment_result = "-"
    emotion_result = "-"
    
    if request.method == 'POST':
        message = request.form['tweet']

        # Sentiment prediction
        sentiment_result = segmentation(AnalysSentiment(message).polarity)

        # Emotion prediction
        emotion_result = predict_emotion(message)

    return render_template("prediction.html",
                           prediction_text=f"{sentiment_result} | Emotion: {emotion_result}")

if __name__ == '__main__':
    app.run(debug=True)
