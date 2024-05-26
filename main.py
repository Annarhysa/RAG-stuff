import wikipediaapi
import faiss
import requests
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent='MyBot/1.0'
)

# Scrape the "Luke Skywalker" Wikipedia page
page = wiki_wiki.page("Luke Skywalker")
text = page.text

# Chunk the text
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

chunks = chunk_text(text)

# Initialize Hugging Face model and tokenizer for embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
model = AutoModel.from_pretrained(embedding_model_name)

# Function to get embeddings from Hugging Face model
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()[0]

# Create FAISS index
dimension = model.config.hidden_size  # Dimension of the embedding vector
index = faiss.IndexFlatL2(dimension)
index_with_ids = faiss.IndexIDMap(index)

# Generate embeddings for chunks and add to FAISS index
embeddings = [get_embedding(chunk) for chunk in chunks]
embeddings_np = np.array(embeddings).astype('float32')
ids = np.array(range(len(chunks)))

index_with_ids.add_with_ids(embeddings_np, ids)

# Initialize Hugging Face pipeline for text generation
qa_model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline('question-answering', model=qa_model_name, tokenizer=qa_model_name)

# Function to query top 3 relevant chunks from FAISS index
def query_faiss(query, k=3):
    query_embedding = get_embedding(query).astype('float32').reshape(1, -1)
    distances, indices = index_with_ids.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

# Function to get answer from Hugging Face model
def get_answer(question, context):
    qa_input = {
        'question': question,
        'context': context
    }
    result = qa_pipeline(qa_input)
    return result['answer']

# Function to handle incoming question from Postman
def handle_question(question):
    relevant_chunks = query_faiss(question)
    context = ' '.join(relevant_chunks)
    answer = get_answer(question, context)
    return answer

# Simulate question from Postman
question = "Who is Luke Skywalker's father?"
answer = handle_question(question)
print(f"Question: {question}\nAnswer: {answer}")

# Flask app to handle HTTP requests (optional)
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    answer = handle_question(question)
    return jsonify({"question": question, "answer": answer})

if __name__ == '__main__':
    app.run(port=5000)