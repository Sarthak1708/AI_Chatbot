# AI_Chatbot

This repository contains an intent-based AI chatbot built using Python, TensorFlow/Keras, and NLTK. The chatbot uses a neural network to understand user input and provide appropriate responses from a predefined knowledge base. The user interface is created with Gradio, making it easy to interact with the bot in a web browser.

## How It Works

The project is divided into two main components: a training script and an application script.

*   **Training (`train_chatbot.py`)**
    1.  Reads conversational intents from the `mpdata1.json` file.
    2.  Uses NLTK to tokenize and lemmatize the text patterns, building a vocabulary of known words (`words.pkl`) and a list of intent classes (`classes.pkl`).
    3.  Converts each sentence into a numerical "bag-of-words" vector.
    4.  Builds and trains a deep learning model (Sequential API in Keras) on the vectorized data.
    5.  Saves the trained model as `chatbot_model.h5`.

*   **Application (`app.py`)**
    1.  Loads the pre-trained model (`chatbot_model.h5`), vocabulary (`words.pkl`), and classes (`classes.pkl`).
    2.  Launches an interactive chat interface using Gradio.
    3.  When a user sends a message, the script processes it into a bag-of-words vector.
    4.  The model predicts the user's intent based on this vector.
    5.  A random response corresponding to the predicted intent is selected from `mpdata1.json` and displayed in the chat.

## Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/sarthak1708/ai_chatbot.git
    cd ai_chatbot
    ```

2.  Install the necessary Python packages:
    ```bash
    pip install tensorflow numpy nltk gradio
    ```
    The scripts will automatically download the required NLTK data (`punkt` and `wordnet`) on first run if they are not found.

## Usage

### Running the Chatbot
To start the chatbot with its Gradio web interface, run the `app.py` script:
```bash
python app.py
```
After the script starts, it will print a local URL (e.g., `http://127.0.0.1:7860`). Open this URL in your web browser to start chatting with the bot.

### Customizing and Re-training
You can customize the chatbot's personality and knowledge by editing the `mpdata1.json` file. After modifying the intents, patterns, or responses, you must re-train the model.

1.  Modify `mpdata1.json` with your custom data.
2.  Run the training script from your terminal:
    ```bash
    python train_chatbot.py
    ```
This will generate new `chatbot_model.h5`, `words.pkl`, and `classes.pkl` files based on your updated data. Once training is complete, you can run `app.py` again to use your updated bot.

## File Structure
```
.
├── app.py              # Main application file with Gradio UI
├── train_chatbot.py    # Script for training the neural network model
├── chatbot_model.h5    # The pre-trained Keras model file
├── mpdata1.json        # The dataset of intents, patterns, and responses
├── words.pkl           # Pickled file containing the vocabulary list
└── classes.pkl         # Pickled file containing the class/tag list
