# app.py
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import gradio as gr

# -------- NLTK SETUP --------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# -------- LOAD FILES & MODEL --------
with open('mpdata1.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

model = load_model('chatbot_model.h5')


# -------- HELPER FUNCTIONS --------
def clean_up_sentence(sentence: str):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence: str):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence: str):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': float(r[1])} for r in results]


def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand. Could you please rephrase?"


# -------- CHATBOT FUNCTION (messages = list of dicts) --------
# chat_history is like: [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, ...]
def chatbot_fn(message, chat_history):
    if chat_history is None:
        chat_history = []

    if not message or not message.strip():
        chat_history.append({
            "role": "assistant",
            "content": "Please type something to start the conversation."
        })
        return "", chat_history

    intents_list = predict_class(message)
    response = get_response(intents_list, intents)

    # User message
    chat_history.append({
        "role": "user",
        "content": message
    })

    # Bot response
    chat_history.append({
        "role": "assistant",
        "content": response
    })

    # Clear textbox input
    return "", chat_history


# -------- CUSTOM CSS --------
custom_css = """
#chatbot-header {
    text-align: center;
    font-family: 'Arial', sans-serif;
    color: #ffffff;
    background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 10px;
}
#chatbot-container {
    background-color: #ffffff;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
    padding: 20px;
    max-width: 800px;
    margin: auto;
}
.chatbot-button {
    background: linear-gradient(90deg, #ff8a00 0%, #da1b60 100%);
    color: white;
    border: none;
    font-size: 16px;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    margin-top: 10px;
}
.chatbot-button:hover {
    background: linear-gradient(90deg, #da1b60 0%, #ff8a00 100%);
}
.chatbot-textbox {
    font-size: 16px;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
}
"""


# -------- BUILD GRADIO UI --------
with gr.Blocks() as demo:
    # Inject CSS
    gr.HTML(f"<style>{custom_css}</style>")

    gr.Markdown(
        """
        <div id="chatbot-header">
            <h1>🌟 AI ChatBot</h1>
            <p>An AI to assist you 24/7</p>
        </div>
        """
    )

    with gr.Group(elem_id="chatbot-container"):
        # No 'type' argument here, but we will still feed dict messages
        chatbot_display = gr.Chatbot(label="Conversation", height=400)
        with gr.Row():
            user_input = gr.Textbox(
                label="Type your message",
                placeholder="Ask me anything...",
                elem_classes=["chatbot-textbox"]
            )
            send_button = gr.Button("Send", elem_classes=["chatbot-button"])
            clear_button = gr.Button("Clear", elem_classes=["chatbot-button"])

        # Send button click
        send_button.click(
            fn=chatbot_fn,
            inputs=[user_input, chatbot_display],
            outputs=[user_input, chatbot_display]
        )

        # Press Enter in textbox
        user_input.submit(
            fn=chatbot_fn,
            inputs=[user_input, chatbot_display],
            outputs=[user_input, chatbot_display]
        )

        # Clear chat
        clear_button.click(
            fn=lambda: [],
            inputs=None,
            outputs=chatbot_display
        )


if __name__ == "__main__":
    demo.launch()
