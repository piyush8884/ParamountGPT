
![Alt Text](https://drive.google.com/uc?id=1wlRgYjZleQmdJvUhR4ePZn1vaKGSifWh)


# Paramount ChatGPT

This is a custom chatbot application that uses LangChain and OpenAI GPT-3.5 to respond to queries based on company-specific data stored in both text and CSV formats. The application is hosted with Flask, with Docker support for easy deployment.

## Features

- Retrieves responses based on company-specific data in text and CSV formats.
- Uses structured metadata to differentiate sources of data.
- Provides feedback storage and downloading functionality.
- Protects the feedback download route with an admin password.
- Allows for similarity calculation on responses.

## Requirements

1. **Python** >= 3.8
2. **Docker** (if you wish to run with Docker)
3. **OpenAI API Key** - You need an OpenAI API key to use GPT-3.5 for answering queries.

## Folder Structure

- `app/`: Contains the application code.
  - `document_processing/`: Contains document loading and splitting logic.
  - `templates/`: Contains HTML templates.
  - `static/`: Contains static files like CSS and JS.
  - `utils/`: Utility functions for profanity checks, feedback handling, etc.
- `dataset/`: Folder where you place your `.txt` and `.csv` files for the chatbot to learn from.
- `.env`: Stores environment variables like OpenAI API key.
- `Dockerfile`: Used to create a Docker image.
- `docker-compose.yml`: Used to orchestrate Docker containers for the application.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Paramount_ChatGPT
```

### 2. Set up the Environment Variables

Create a `.env` file in the root directory and add the following line:

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

### 3. Install Dependencies

Create a virtual environment and activate it:

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place your `.txt` and `.csv` files in the `dataset/` folder. The loader will process these files to build the knowledge base for the chatbot.

### 5. Run the Application

To run the Flask application, use:

```bash
python app.py
```

The server should now be running at `http://localhost:5000`.

### 6. Docker Setup (Optional)

If you prefer to run the application in a Docker container:

1. **Build the Docker image:**

   ```bash
   docker build -t paramount-chatgpt .
   ```

2. **Run the Docker container using docker-compose:**

   ```bash
   docker-compose up
   ```

This will start the application on `http://localhost:5000`.

## Usage

### Querying the Chatbot

Send a POST request to `/ask` with a JSON body containing your question:

```json
{
  "question": "What is the mission of Paramount?"
}
```

### Downloading Feedback (Admin Only)

To download feedback, access the `/download_feedback` endpoint:

```bash
http://localhost:5000/download_feedback?password=superadminpassword
```

Replace `superadminpassword` with your admin password.

### Example Queries

- "founders of paramount"

## Contributing

Feel free to fork this repository and submit pull requests.
