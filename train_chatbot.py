# train_chatbot.py

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

# Make sure required NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# ----- Load intents JSON -----
# Make sure mpdata1.json is in the same folder as this script
with open('mpdata1.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# ----- Build vocabulary and documents -----
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower-case, remove punctuation-like symbols
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

# Save words and classes for later use
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# ----- Prepare training data -----
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for w in words:
        bag.append(1 if w in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training, dtype=np.float32)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# ----- Build the model -----
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(train_x[0]),)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("Training the model...")
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# ----- Save the model -----
model.save('chatbot_model.h5')
print("Training complete. Model saved as chatbot_model.h5")
