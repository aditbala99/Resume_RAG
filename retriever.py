retriever 
from typing import List
import numpy as np
import os
import requests
from requests.exceptions import RequestException
from fastapi import FastAPI
from collections import defaultdict
from pydantic import BaseModel
import uvicorn
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from retrievers.document_query_retriever import DocumentQueryRetriever
from retrievers.document_topk_retriever import DocumentTopKRetriever
from retrievers.multi_query_retriever import MultiQueryRetriever
from retrievers.simple_combiner import SimpleCombiner
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings   
from utils.document_handling import Document_to_dict, dict_to_Document
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import time
from utils.s3 import get_all_pdfs_from_s3
from dotenv import load_dotenv
from langchain_cohere import CohereRerank

from utils.document_handling import list_files_with_full_paths

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from parsers.llamaparse import llamaparse_with_cache
#from langchain.logging import LangSmithLogger
# vector stores
#from langchain.vectorstores import Chroma
#from langchain_qdrant import Qdrant
from langchain_community.vectorstores import FAISS

import logging
logging.basicConfig(filename='retriever.log', level=logging.DEBUG, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize logger
logger = logging.getLogger(_name_)
##langsmith_logger = LangSmithLogger(log_to_file=True, log_file_path="langsmith.log")

import os 
os.environ["COHERE_API_KEY"] = "ghe9rUi2U5HRq5S9KnQEMUgPuwjO1GVnoSGSK79W"

BUCKET_NAME = "cogniq-org01-dev"
CACHE_FOLDER = "pymupdf_cache"
DATASTORE = "s3"
SUMMARY_LIMIT = 10

# Define the path to the vector database
VECTOR_DB_PATH ="chroma"
# Define FastAPI app
app = FastAPI()

# Define a request model for document embedding
class Texts(BaseModel):
    texts: List[str]

# Define a request model for query embedding
class Query(BaseModel):
    text: str


EMBEDDING_ENDPOINT ="http://0.0.0.0:5002" 
PARSER_ENDPOINT = "http://0.0.0.0:3002"
dotenv_path = '../.env'
load_dotenv(dotenv_path=dotenv_path)

def _post_request( url, payload):
        """
        Helper method to make a POST request and handle potential errors.
        """
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses
            return response.json()
        except RequestException as e:
            logging.error(f"Request to {url} with payload {payload} failed: {e}")
            raise
        except ValueError as e:
            logging.error(f"Invalid JSON response from {url}: {e}")
            raise
    
class PDFParseRequest(BaseModel):
    pdf_files: List[str]

class Embedding_API(Embeddings):
    """
    This class represents an API for embedding documents and queries using a retriever.
    It inherits from the Embeddings class.
    """

    def _init_(self, embedding_endpoint):
        """
        Initializes an instance of the Embedding_API class.

        Args:
            retriever: An instance of the retriever used for embedding documents and queries.
        """
        self.embedding_endpoint = embedding_endpoint
    

        def embed_documents(self, documents):
        """
        Call API to get document embeddings.
        """
        t1 = time.time()
        url = f"{self.embedding_endpoint}/embed_documents/"
        try:
            response = _post_request(url, {"texts": documents})
            logging.info(f"Document embeddings retrieved successfully")
            t2 = time.time()
            elapsed_time = t2 - t1
            print(f"{t2-t1} time taken for embedding the documents")
            logging.info(f"Time taken for embedding documents: {t2-t1}seconds")
            return response
        except Exception as e:
            logging.error(f"Embed documents failed: {e}")
            raise

    def embed_query(self, query):
        """
        Call API to get query embedding.
        """
        t1 = time.time()
        url = f"{self.embedding_endpoint}/embed_query/"
        try:
            response = _post_request(url, {"text": query})
            logging.info(f"Query embedding retrieved successfully")
            t2 = time.time()
            elapsed_time = t2 - t1
            print("elapsed time for embedding query", elapsed_time)
            logging.info(f"Time taken for embedding query: {elapsed_time} seconds")
            return response
        except Exception as e:
            logging.error(f"Embed query failed for query {query}: {e}")
            raise



class Retriever():
    def _init_(self, retriever, embeddings_model, chat_llm, parser_endpoint, embedding_endpoint):
        """
        Initializes an instance of the Retriever class.

        Args:
            retriever: The retriever object.
            chat_llm: The chat language model object.
            parser_endpoint: The endpoint for parsing documents.
            embedding_endpoint: The endpoint for embedding documents and queries.
        """
        self.retriever = retriever
        self.chat_llm = chat_llm
        self.parser_endpoint = parser_endpoint
        self.embedding_endpoint = embedding_endpoint
        self.embeddings_model = embeddings_model
    
    def set_retriver(self, retriever):
        """
        Sets the retriever object.

        Args:
            retriever: The retriever object.
        """
        self.retriever = retriever

    def parse_documents(self, file_paths):
        """
        Call API to parse documents.
        """
        t1 = time.time()
        url = f"{self.parser_endpoint}/parse_pdfs/"
        payload = {
            "pdf_files": file_paths,  # List of PDF files
            "bucket_name": "cogniq-org01-dev",
            "cache_folder": "pymupdf_cache",
            "datastore": "s3",
            "summary_limit": 10
        }

        try:
            response = _post_request(url, payload)
            logging.info(f"Documents parsed successfully")
            #print(f"{t2-t1} time taken for parsing the documents")
            t2 = time.time()
            elapsed_time = t2 - t1
            print(f"{t2-t1} time taken for parsing the documents")
            logging.info(f"Time taken for parsing documents: {elapsed_time:.2f} seconds")
            return response
        except Exception as e:
            #(f"Parse documents failed for {file_paths}: {e}")
            raise
        print(f"{t2-t1} time taken for parsing the documents")
        t2 = time.time()
        elapsed_time = t2 - t1
        print(f"{t2-t1} time taken for parsing the documents")
        logging.info(f"Time taken for parsing documents: {elapsed_time:.2f} seconds")

    def setup_document_chain(self, document, global_retriever, embeddings_model, primary_qa_llm):
        """
        Sets up a document chain for retrieval.

        Args:
            document: The document to be used in the chain.
            global_retriever: The global retriever object.
            embeddings_model: The embeddings model object.
            primary_qa_llm: The primary question-answering language model object.

        Returns:
            The query retriever object.
        """
        query_retriever = DocumentQueryRetriever.from_llm(retriever=global_retriever, llm=primary_qa_llm, documents=document)
        return query_retriever

    def setup_document_chain(self, document: str, global_retriever: MultiQueryRetriever, embeddings_model: OpenAIEmbeddings, primary_qa_llm: ChatOpenAI) -> DocumentQueryRetriever:
        query_retriever = DocumentQueryRetriever.from_llm(retriever=global_retriever, llm=primary_qa_llm, documents=document)
        return query_retriever


     
    def setup_chain(self, pdf_files: List[str]):
        print(f'Initial setup chain for {pdf_files}')
        documents = retriever.parse_documents(pdf_files)['documents']
        # Generate embeddings and create FAISS index this calls the embedding service
        documents, documents_dict, page_lookups = dict_to_Document(documents) 
        # Generate embeddings and create FAISS index
        vector_store = FAISS.from_documents(documents, self.embeddings_model)
        primary_qa_llm = self.chat_llm
        global_retriever = vector_store.as_retriever()
        #set up multi query retriever
        multi_query_retriever = MultiQueryRetriever.from_llm(retriever=global_retriever, llm=primary_qa_llm, documents = documents)
        document_retrievers = [multi_query_retriever]
        for file, document in documents_dict.items():
            document_retrievers.append(self.setup_document_chain(document, global_retriever, embeddings_model, primary_qa_llm))
        #Use combiner to combine retrievers
        simple_combiner = SimpleCombiner(retrievers=document_retrievers)
        compressor = CohereRerank(top_n = None) 
        #Reranker
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=simple_combiner, base_compressor=compressor
        )
        #Get top k chunks from each document also attaches chunks below the selected chunk for completeness
        topk_retriever = DocumentTopKRetriever(
                base_retriever=compression_retriever, topk=2, page_lookup=page_lookups
        ) 
        self.retriever = topk_retriever


##############################################################################################################
# Define the retriever object
embeddings_model = Embedding_API(embedding_endpoint=EMBEDDING_ENDPOINT)
retriever = Retriever(retriever = None, embeddings_model = embeddings_model, chat_llm = ChatOpenAI(model_name="gpt-4o", temperature=0), parser_endpoint=PARSER_ENDPOINT, embedding_endpoint=EMBEDDING_ENDPOINT)
pdf_files = get_all_pdfs_from_s3(bucket_name = BUCKET_NAME, folder = 'data') 
retriever.setup_chain(pdf_files)

@app.post("/send_files/", response_model=str)
def setup_retrieval_chain(request: PDFParseRequest):
    """
    Sets up the retrieval chain for the parsed documents.

    Args:
        request: The request containing the PDF file paths.

    Returns:
        A string indicating the success of the operation.
    """
    file_paths = request.pdf_files
    print(f'Calling setup chain for {file_paths}')
    retriever.setup_chain(file_paths)
  
    return "Parsed documents"

@app.post("/get_relevant_documents/", response_model=List[dict])
def retrieve_documents(query: Query):
    """
    Retrieves relevant documents based on the given query.

    Args:
        query: The query to retrieve relevant documents.

    Returns:
        A list of relevant documents in dictionary format.
    """
    # Log the retrieval query
    #langsmith_logger.log_event("RetrievalQuery", query=query.text)
    #Get relevant documents 
    documents = retriever.retriever.get_relevant_documents(query.text)

    #Convert documents to dictionary format which is serializable
    document_dicts = Document_to_dict(documents)
    for document_dict in document_dicts:
        if 'relevance_score' in document_dict['metadata']:
            # Convert the relevance score to a string
            if 'relevance_score' in document_dict['metadata']:
                document_dict['metadata']['relevance_score'] = str(document_dict['metadata']['relevance_score'])
    
    #langsmith_logger.save_logs()
    return document_dicts

if _name_ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=4002)
