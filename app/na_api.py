# import os
# from flask import Flask, request, jsonify, render_template, send_file
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
# from langchain.agents import create_csv_agent
# from langchain.llms import OpenAI
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Load environment variables
# load_dotenv()
#
# app = Flask(__name__)
# CORS(app)
#
# # Configuration
# app.config['MAX_CONTENT_LENGTH'] = 0.2 * 1024 * 1024  # 200KB limit for uploads
# UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#
# # OpenAI API Key
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY environment variable is not set.")
#
# try:
#     # Load and split dataset for document-based QA
#     documents = load_documents('./dataset')
#     texts = split_documents(documents)
#
#     # Initialize vector store and retriever
#     embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
#     persist_directory = 'db'
#     vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
#     retriever = vectordb.as_retriever(search_kwargs={"k": 5})
#
#     # Prompt Template
#     prompt_template = """
#     You are an AI assistant with access to specific information. Answer the question based only on the following retrieved content:
#     ---
#     {context}
#     ---
#     Question: {question}
#
#     Answer in a conversational style without adding any information not found in the context.
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#
#     # Set up RetrievalQA with custom prompt
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt}
#     )
# except Exception as e:
#     raise RuntimeError(f"Error initializing document-based QA: {e}")
#
#
# def calculate_similarity(query, response):
#     try:
#         vectorizer = TfidfVectorizer()
#         tfidf_query = vectorizer.fit_transform([query])
#         tfidf_response = vectorizer.transform([response])
#         similarity = cosine_similarity(tfidf_query, tfidf_response)
#         return similarity[0][0]
#     except Exception as e:
#         print(f"Error calculating similarity: {e}")
#         return 0  # Return a default similarity of 0 if error occurs
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/ask', methods=['POST'])
# def ask():
#     query = request.json.get('question')
#     if not query:
#         return jsonify({'answer': 'Question cannot be empty.', 'similarity': 0, 'status': 400})
#
#     # Check for profanity
#     if contains_profanity(query):
#         return jsonify({'answer': 'Inappropriate language detected.', 'similarity': 0, 'status': 400})
#
#     try:
#         # Retrieve and format response with source documents
#         result = qa_chain({"query": query})
#         answer = result['result']
#         source_documents = result.get('source_documents', [])
#         source_texts = [doc.page_content for doc in source_documents] if source_documents else []
#
#         # Calculate similarity score
#         similarity = calculate_similarity(query, answer) if answer else 0
#         status = 204 if "I do not have information" in answer or similarity < 0.8 else 200
#
#         return jsonify({
#             'answer': answer,
#             'similarity': similarity,
#             'status': status,
#             'source_documents': source_texts
#         })
#     except Exception as e:
#         return jsonify({'error': f"An error occurred while processing the question: {e}", 'status': 500})
#
#
# @app.route('/feedback', methods=['POST'])
# def feedback():
#     feedback_data = request.json
#     query = feedback_data.get("query")
#     response = feedback_data.get("response")
#     rating = feedback_data.get("rating")
#
#     if not query or not response or rating is None:
#         return jsonify({"error": "Feedback data is incomplete."}), 400
#
#     try:
#         feedback_entry = {"query": query, "response": response, "rating": rating}
#         store_feedback(feedback_entry)
#         return jsonify({"message": "Feedback saved successfully."})
#     except Exception as e:
#         return jsonify({"error": f"Failed to save feedback: {e}"}), 500
#
#
# @app.route('/download_feedback', methods=['GET'])
# def download_feedback():
#     admin_password = request.args.get("password")
#     if admin_password != "superadminpassword":
#         return jsonify({"error": "Unauthorized access"}), 403
#
#     feedback_file = os.path.join(os.getcwd(), "feedback.csv")
#     if os.path.isfile(feedback_file):
#         try:
#             return send_file(feedback_file, as_attachment=True)
#         except Exception as e:
#             return jsonify({"error": f"Error sending feedback file: {e}"}), 500
#     else:
#         return jsonify({"error": "Feedback file not found. Please wait to download."}), 404
#
#
# @app.route('/chat_csv', methods=['POST'])
# def chat_csv():
#     if openai_api_key is None or openai_api_key == "":
#         return jsonify({"error": "OPENAI_API_KEY is not set"}), 500
#
#     query = request.form.get('query')
#     csv_file = request.files.get('csv_file')
#     if not query:
#         return jsonify({"error": "No user question provided"}), 400
#     if csv_file is None:
#         return jsonify({"error": "No CSV file provided"}), 400
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
#         return jsonify({"answer": response}), 200
#     except Exception as e:
#         return jsonify({"error": f"Failed to process CSV file or run query: {e}"}), 500
#     # finally:
#     #     # Optional cleanup to delete uploaded files after processing
#     #     if os.path.exists(file_path):
#     #         os.remove(file_path)
#
#
# if __name__ == '__main__':
#     try:
#         app.run(host='0.0.0.0', port=5000, debug=True)
#     except Exception as e:
#         print(f"Failed to start the server: {e}")



# ALWAYS ON THE REDIS SERVER EXE



import os
import logging
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_redis import FlaskRedis
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
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

# Specify the allowed origins

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5500", "http://192.168.0.48"]}})


# Configure Redis
app.config['REDIS_URL'] = "redis://localhost:6379/0"
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
        csv_file.save(file_path)
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
    finally:
        # Optional cleanup to delete uploaded files after processing
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Uploaded CSV file removed after processing.")


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

