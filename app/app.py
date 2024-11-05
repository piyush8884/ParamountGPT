# import os
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from dotenv import load_dotenv
# from app.document_processing.loader import load_documents
# from app.document_processing.splitter import split_documents
# from app.utils.profanity_check import contains_profanity
# from app.utils.feedback_handler import store_feedback
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from flask import send_file
# # Load environment variables
# load_dotenv()
#
# app = Flask(__name__)
# CORS(app)
#
# # Set OpenAI API key
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY environment variable is not set.")
#
# # Load and split dataset
# documents = load_documents('./dataset')
# texts = split_documents(documents)
#
# # Initialize vector store and retriever
# embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
# persist_directory = 'db'
# vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
# retriever = vectordb.as_retriever(search_kwargs={"k": 5})
#
# # Custom prompt template to restrict LLM to only use retrieved content
# prompt_template = """
# You are an AI assistant with access to specific information. Answer the question based only on the following retrieved content:
# ---
# {context}
# ---
# Question: {question}
#
# Answer in a conversational style without adding any information not found in the context.
# """
#
# prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#
# # Set up RetrievalQA with custom prompt
# qa_chain = RetrievalQA.from_chain_type(
#     llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": prompt}
# )
#
#
# def calculate_similarity(query, response):
#     vectorizer = TfidfVectorizer()
#     tfidf_query = vectorizer.fit_transform([query])
#     tfidf_response = vectorizer.transform([response])
#     similarity = cosine_similarity(tfidf_query, tfidf_response)
#     return similarity[0][0]
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/ask', methods=['POST'])
# def ask():
#     query = request.json['question']
#
#     # Check for profanity
#     if contains_profanity(query):
#         return jsonify({'answer': 'Inappropriate language detected.', 'similarity': 0, 'status': 400})
#
#     try:
#         # Retrieve and format response with source documents
#         result = qa_chain({"query": query})
#         answer = result['result']
#         source_documents = result['source_documents']
#
#         # Print source documents for debugging
#         # for idx, doc in enumerate(source_documents):
#         #     print(f"Source Document {idx + 1}:\n{doc.page_content}\n")
#
#         # Include source documents in the response
#         source_texts = [doc.page_content for doc in source_documents]
#
#         # Calculate similarity score (optional)
#         similarity = calculate_similarity(query, answer) if answer else 0
#
#         # Additional check: If answer is generic or not relevant to dataset, set status to 204
#         if "I do not have information" in answer or similarity < 0.8:
#             status = 204
#         else:
#             status = 200
#
#         return jsonify({
#             'answer': answer,
#             'similarity': similarity,
#             'status': status,
#             'source_documents': source_texts  # Include source documents in response
#         })
#
#     except Exception as e:
#         return jsonify({'error': str(e), 'status': 500})
#
#
# @app.route('/feedback', methods=['POST'])
# def feedback():
#     feedback_data = request.json
#     query = feedback_data.get("query")
#     response = feedback_data.get("response")
#     rating = feedback_data.get("rating")
#
#     # Save feedback using store_feedback function
#     feedback_entry = {"query": query, "response": response, "rating": rating}
#     store_feedback(feedback_entry)
#
#     return jsonify({"message": "Feedback saved successfully."})
#
#
# @app.route('/download_feedback', methods=['GET'])
# def download_feedback():
#     # Replace 'superadminpassword' with a secure password or authentication mechanism
#     admin_password = request.args.get("password")
#     if admin_password != "superadminpassword":
#         return jsonify({"error": "Unauthorized access"}), 403
#
#     # Explicit path to feedback.csv file
#     feedback_file = os.path.join(os.getcwd(), "feedback.csv")
#
#     # Logging to check if the route is accessed multiple times
#     print("Attempting to download feedback.csv...")
#     print("Feedback file path:", feedback_file)
#
#     # Check if feedback.csv exists before sending
#     if os.path.isfile(feedback_file):
#         print("File found. Downloading...")
#         return send_file(feedback_file, as_attachment=True)
#     else:
#         print("File not found.")
#         return jsonify({"error": "Feedback file found.Please wait to download."}), 404
#
# # http://localhost:5000/download_feedback?password=superadminpassword
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)





import os
import logging
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_redis import FlaskRedis
from dotenv import load_dotenv
# from app.document_processing.loader import load_documents
# from app.document_processing.splitter import split_documents
# from app.utils.profanity_check import contains_profanity
# from app.utils.feedback_handler import store_feedback
#this three for production
from document_processing.loader import load_documents
from document_processing.splitter import split_documents
from utils.profanity_check import contains_profanity
from utils.feedback_handler import store_feedback
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from langchain.prompts import PromptTemplate
# Specify the allowed origins

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)


# Configure Redis
app.config['REDIS_URL'] = "redis://redis:6379/0"
#localrunning
# app.config['REDIS_URL'] = "redis://localhost:6379/0"
redis_client = FlaskRedis(app)

# Logging configuration
LOG_FOLDER = os.path.join(app.root_path, 'Logs')
os.makedirs(LOG_FOLDER, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_FOLDER, 'app.log'), level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 0.2 * 1024 * 1024  # 200KB limit for uploads
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable is not set.")
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

try:
    # Load and split dataset for document-based QA
    documents = load_documents('./dataset')
    texts = split_documents(documents)

    # Initialize vector store and retriever
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # Prompt Template
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
except Exception as e:
    logger.error(f"Error initializing document-based QA: {e}")
    raise RuntimeError(f"Error initializing document-based QA: {e}")


def calculate_similarity(query, response):
    try:
        vectorizer = TfidfVectorizer()
        tfidf_query = vectorizer.fit_transform([query])
        tfidf_response = vectorizer.transform([response])
        similarity = cosine_similarity(tfidf_query, tfidf_response)
        return similarity[0][0]
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0  # Return a default similarity of 0 if error occurs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('question')
    if not query:
        return jsonify({'answer': 'Question cannot be empty.', 'similarity': 0, 'status': 400})

    # Check for profanity
    if contains_profanity(query):
        return jsonify({'answer': 'Inappropriate language detected.', 'similarity': 0, 'status': 400})

    # Check Redis cache for cached response
    cached_answer = redis_client.get(f"ask:{query}")
    if cached_answer:
        logger.info("Returning cached answer for ask endpoint.")
        return jsonify({'answer': cached_answer.decode('utf-8'), 'status': 200})

    try:
        # Retrieve and format response with source documents
        result = qa_chain({"query": query})
        answer = result['result']
        source_documents = result.get('source_documents', [])
        source_texts = [doc.page_content for doc in source_documents] if source_documents else []

        # Calculate similarity score
        similarity = calculate_similarity(query, answer) if answer else 0
        status = 204 if "I do not have information" in answer or similarity < 0.8 else 200

        # Cache the result in Redis
        redis_client.set(f"ask:{query}", answer)

        return jsonify({
            'answer': answer,
            'similarity': similarity,
            'status': status,
            'source_documents': source_texts
        })
    except Exception as e:
        logger.error(f"An error occurred while processing the question: {e}")
        return jsonify({'error': f"An error occurred while processing the question: {e}", 'status': 500})

#
# @app.route('/chat_csv', methods=['POST'])
# def chat_csv():
#     if openai_api_key is None or openai_api_key == "":
#         logger.error("OPENAI_API_KEY is not set.")
#         return jsonify({"error": "OPENAI_API_KEY is not set"}), 500
#
#     query = request.form.get('query')
#     csv_file = request.files.get('csv_file')
#     if not query:
#         return jsonify({"error": "No user question provided"}), 400
#     if csv_file is None:
#         return jsonify({"error": "No CSV file provided"}), 400
#
#     # Check Redis cache for cached response
#     cached_response = redis_client.get(f"csv:{query}")
#     if cached_response:
#         logger.info("Returning cached answer for chat_csv endpoint.")
#         return jsonify({"answer": cached_response.decode('utf-8')}), 200
#
#     original_filename = csv_file.filename
#     file_path = os.path.join(UPLOAD_FOLDER, original_filename)
#
#     try:
#         csv_file.save(file_path)
#         agent = create_csv_agent(
#             ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500),
#             file_path,
#             verbose=True
#         )
#         response = agent.run(query)
#
#         # Cache the CSV response in Redis
#         redis_client.set(f"csv:{query}", response)
#
#         return jsonify({"answer": response}), 200
#     except Exception as e:
#         logger.error(f"Failed to process CSV file or run query: {e}")
#         return jsonify({"error": f"Failed to process CSV file or run query: {e}"}), 500




## suugestion based
@app.route('/chat_csv', methods=['POST'])
def chat_csv():
    if openai_api_key is None or openai_api_key == "":
        logger.error("OPENAI_API_KEY is not set.")
        return jsonify({"error": "OPENAI_API_KEY is not set"}), 500

    query = request.form.get('query')
    csv_file = request.files.get('csv_file')
    if not query:
        return jsonify({"error": "No user question provided"}), 400
    if csv_file is None:
        return jsonify({"error": "No CSV file provided"}), 400

    # Check Redis cache for cached response
    cached_response = redis_client.get(f"csv:{query}")
    if cached_response:
        logger.info("Returning cached answer for chat_csv endpoint.")
        return jsonify({"answer": cached_response.decode('utf-8')}), 200

    original_filename = csv_file.filename
    file_path = os.path.join(UPLOAD_FOLDER, original_filename)

    try:
        # Save the CSV file temporarily
        csv_file.save(file_path)

        # Check if the query is asking for suggestions or improvements
        if any(word in query.lower() for word in ["suggestions", "improvements", "recommendations"]):
            # Use OpenAI model to generate suggestions based on CSV data context
            response = generate_contextual_suggestions(file_path, query)
        else:
            # Handle as a question-based query
            agent = create_csv_agent(
                ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=500),
                file_path,
                verbose=True
            )
            response = agent.run(query)

        # Cache the CSV response in Redis
        redis_client.set(f"csv:{query}", response)

        return jsonify({"answer": response}), 200
    except Exception as e:
        logger.error(f"Failed to process CSV file or run query: {e}")
        return jsonify({"error": f"Failed to process CSV file or run query: {e}"}), 500

def generate_contextual_suggestions(file_path, query):
    """
    Uses the language model to generate suggestions based on CSV data content in a specific business context.
    """
    import pandas as pd

    try:
        # Load the CSV file data
        data = pd.read_csv(file_path)

        # Use the OpenAI model to generate suggestions specific to the query context
        prompt_template = """
        You are an educational analyst helping to improve student performance based on their marks data. Analyze the provided data on student marks and provide detailed, practical suggestions for improvement.

        Consider:
        - Identifying patterns or weaknesses in student performance.
        - Recommending study strategies, resources, or activities that would help improve marks.
        - Suggesting areas where additional tutoring or support may be beneficial.
        - Offering general advice on how to approach subjects or topics where students are struggling.

        Request:
        {query}

        Data Preview:
        {data_preview}

        Provide actionable and detailed recommendations to improve student marks.
        """
        data_preview = data.head().to_string()  # Provide a small preview of the data
        prompt = prompt_template.format(data_preview=data_preview, query=query)

        # Use the language model to generate a response
        chat_model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
        response = chat_model({"prompt": prompt})["text"]

        return response
    except Exception as e:
        logger.error(f"Failed to generate suggestions: {e}")
        return "Failed to generate suggestions due to an error."




@app.route('/download_feedback', methods=['GET'])
def download_feedback():
    admin_password = request.args.get("password")
    if admin_password != "superadminpassword":
        logger.warning("Unauthorized feedback download attempt.")
        return jsonify({"error": "Unauthorized access"}), 403

    feedback_file = os.path.join(os.getcwd(), "feedback.csv")
    if os.path.isfile(feedback_file):
        try:
            return send_file(feedback_file, as_attachment=True)
        except Exception as e:
            logger.error(f"Error sending feedback file: {e}")
            return jsonify({"error": f"Error sending feedback file: {e}"}), 500
    else:
        logger.warning("Feedback file not found.")
        return jsonify({"error": "Feedback file not found. Please wait to download."}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

