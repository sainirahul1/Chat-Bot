from flask import Flask, render_template, request, jsonify
from nltk_utils import tokenize, bag_of_words
from model import NeuralNet
import torch
import json
import random

app = Flask(__name__)

# Load intents
with open("intents.json", "r", encoding="utf-8-sig") as f:
    intents = json.load(f)


# Load model
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sentence = data["message"]
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return jsonify({"answer": random.choice(intent["responses"])})
    return jsonify({"answer": "I'm not sure I understand. Can you rephrase?"})

if __name__ == "__main__":
    print("✅ Flask is running at http://127.0.0.1:5000")
    app.run(debug=True)
