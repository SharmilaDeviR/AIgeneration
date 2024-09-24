from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the pre-trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

def generate_question(answer, context):
    input_text = f"answer: {answer}  context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    outputs = model.generate(input_ids)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

# Example usage
answer = "Paris"
context = "Paris is the capital of France."
question = generate_question(answer, context)
print(question)
