# from langchain.document_loaders import DirectoryLoader, TextLoader
# from langchain.schema import Document
# import os
# import pandas as pd
#
# def load_documents(directory):
#     loader = DirectoryLoader(directory, glob="*.txt", loader_cls=TextLoader)
#     return loader.load()

# # app/dments    return documents
#















from langchain.schema import Document
import os
import pandas as pd


def load_documents(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()

            if file_extension == ".txt":
                # Load .txt files
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append(Document(page_content=content, metadata={"source": file_path}))

            elif file_extension == ".csv":
                # Load .csv files
                df = pd.read_csv(file_path)
                content = df.to_string(index=False)  # Convert DataFrame to a string
                documents.append(Document(page_content=content, metadata={"source": file_path}))

            elif file_extension in [".xls", ".xlsx"]:
                # Load .xlsx files
                df = pd.read_excel(file_path)
                content = df.to_string(index=False)  # Convert DataFrame to a string
                documents.append(Document(page_content=content, metadata={"source": file_path}))

            else:
                print(f"Unsupported file type: {file}")

    return documents
