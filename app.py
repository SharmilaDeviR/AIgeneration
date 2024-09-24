from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import tensorflow as tf

app = Flask(__name__)

# Load models
ai_model = tf.keras.models.load_model('path/to/ai_model.h5')
qg_model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = ai_model.predict(data['input'])
    return jsonify({'prediction': prediction.tolist()})

@app.route('/generate_question', methods=['POST'])
def generate_question():
    data = request.json
    input_text = f"answer: {data['answer']}  context: {data['context']}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = qg_model.generate(input_ids)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'question': question})

if __name__ == '__main__':
    app.run(debug=True)
