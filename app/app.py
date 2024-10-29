import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from app.document_processing.loader import load_documents
from app.document_processing.splitter import split_documents
from app.utils.profanity_check import contains_profanity
from app.utils.feedback_handler import store_feedback
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import send_file
# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Load and split dataset
documents = load_documents('./dataset')
texts = split_documents(documents)

# Initialize vector store and retriever
embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Custom prompt template to restrict LLM to only use retrieved content
prompt_template = """
You are an AI assistant with access to specific information. Answer the question based only on the following retrieved content:
---
{context}
---
Question: {question}

Answer in a conversational style without adding any information not found in the context.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Set up RetrievalQA with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


def calculate_similarity(query, response):
    vectorizer = TfidfVectorizer()
    tfidf_query = vectorizer.fit_transform([query])
    tfidf_response = vectorizer.transform([response])
    similarity = cosine_similarity(tfidf_query, tfidf_response)
    return similarity[0][0]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    query = request.json['question']

    # Check for profanity
    if contains_profanity(query):
        return jsonify({'answer': 'Inappropriate language detected.', 'similarity': 0, 'status': 400})

    try:
        # Retrieve and format response with source documents
        result = qa_chain({"query": query})
        answer = result['result']
        source_documents = result['source_documents']

        # Print source documents for debugging
        # for idx, doc in enumerate(source_documents):
        #     print(f"Source Document {idx + 1}:\n{doc.page_content}\n")

        # Include source documents in the response
        source_texts = [doc.page_content for doc in source_documents]

        # Calculate similarity score (optional)
        similarity = calculate_similarity(query, answer) if answer else 0

        # Additional check: If answer is generic or not relevant to dataset, set status to 204
        if "I do not have information" in answer or similarity < 0.8:
            status = 204
        else:
            status = 200

        return jsonify({
            'answer': answer,
            'similarity': similarity,
            'status': status,
            'source_documents': source_texts  # Include source documents in response
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 500})


@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_data = request.json
    query = feedback_data.get("query")
    response = feedback_data.get("response")
    rating = feedback_data.get("rating")

    # Save feedback using store_feedback function
    feedback_entry = {"query": query, "response": response, "rating": rating}
    store_feedback(feedback_entry)

    return jsonify({"message": "Feedback saved successfully."})


@app.route('/download_feedback', methods=['GET'])
def download_feedback():
    # Replace 'superadminpassword' with a secure password or authentication mechanism
    admin_password = request.args.get("password")
    if admin_password != "superadminpassword":
        return jsonify({"error": "Unauthorized access"}), 403

    # Explicit path to feedback.csv file
    feedback_file = os.path.join(os.getcwd(), "feedback.csv")

    # Logging to check if the route is accessed multiple times
    print("Attempting to download feedback.csv...")
    print("Feedback file path:", feedback_file)

    # Check if feedback.csv exists before sending
    if os.path.isfile(feedback_file):
        print("File found. Downloading...")
        return send_file(feedback_file, as_attachment=True)
    else:
        print("File not found.")
        return jsonify({"error": "Feedback file found.Please wait to download."}), 404

# http://localhost:5000/download_feedback?password=superadminpassword

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
