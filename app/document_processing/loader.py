from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
import os
import pandas as pd

def load_documents(directory):
    loader = DirectoryLoader(directory, glob="*.txt", loader_cls=TextLoader)
    return loader.load()

# app/dments    return documents

