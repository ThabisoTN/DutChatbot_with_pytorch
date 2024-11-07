import random
import json
import torch
from model import NeuralNet
from flask import Flask, render_template, request, jsonify
from nltk_utils import bag_Of_words, tokenize
import nltk
nltk.data.path.append('./nltk_data')  # Adjust the path if necessary


app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('intents.json', 'r') as f:
    intents = json.load(f)

file = "data.pth"
data = torch.load(file)

input_size = data["input_size"]
hidden_size = data["hiddent_size"]  
output_size = data["output_size"]
all_words = data["all_word"]  
tags = data["tags"]
model_state = data["model_state"]


model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Thabiso_Dut"

@app.route("/")
def index():
    return render_template("index.html")

# Process user message and return chatbot response
@app.route("/get-response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    
    sentence = tokenize(user_input)
    x = bag_Of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return jsonify({"response": random.choice(intent['responses'])})
    return jsonify({"response": "Am not trained to talk about that and am only trained to talked about DUT related information!!!."})

# If running directly, use Flask for the web interface
if __name__ == "__main__":
    print("Starting web server...")
    app.run(debug=True)
