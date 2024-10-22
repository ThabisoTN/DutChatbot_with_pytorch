# Thabiso_Dut Chatbot with PyTorch

This project is a simple chatbot built using PyTorch and Flask. The chatbot leverages a neural network model for natural language processing, and it is served through a web interface. Users can interact with the chatbot, which is capable of responding to various user queries based on predefined intents stored in a JSON file.

## Features
- Trained neural network chatbot using PyTorch.
- Integrated with Flask to serve a web interface.
- Responsive web chat UI where users can type messages and receive bot responses.
- Handles various user queries as defined in `intents.json`.

## Prerequisites

Before running this project, ensure you have the following installed:
- Python 3.6+
- Flask
- PyTorch
- NLTK (for tokenizing words)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/DutChatbot_with_pytorch.git
   cd DutChatbot_with_pytorch


2. python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. pip install -r requirements.txt


4. import nltk
nltk.download('punkt')


5. python train.py(Train the bot)


6. python chat.py(Run the chatbot on local server)

