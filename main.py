main.py 
## Author: Aditya 
### Main Functionality: Take in the users input from /chat endpoint, send the input to the retriever service to get relevant documents, and then stream the response from the LLM

import os
import time
import pandas as pd
import requests
from typing import List
import boto3
import json
import fitz 
import logging
from operator import itemgetter
from enum import Enum
import re

# FastAPI imports
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from fuzzysearch import find_near_matches

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from llama_index.core.node_parser import SentenceSplitter

from utils.dataframe_handling import add_indices
from utils.document_handling import dict_to_Document_simple, Document_to_dict, format_docs_with_id
from dotenv import load_dotenv
from utils.langchain import AnnotatedAnswer


# Configure logging
logging.basicConfig( filename='main.log', level=logging.DEBUG,filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(_name_)

# Ensure the logger has handlers
if not logger.handlers:
    file_handler = logging.FileHandler('main.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

# Initialize FastAPI app
app = FastAPI()
logger.info("FastAPI app initialized")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or specify your allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# S3 bucket name
BUCKET_NAME = 'your bucket name'
s3_client = boto3.client('s3')

# Define the endpoint for the retriever service
RETRIEVER_ENDPOINT = "http://0.0.0.0:4000"
SIMDIFF_ENDPOINT = "http://0.0.0.0:8000"

# Set APIs keys for different services
dotenv_path = '../.env'
load_dotenv(dotenv_path=dotenv_path)

class CitationStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NOT_FOUND = "not_found"

# Define the RAG class
class RAG():
    def _init_(self, model_name='gpt-4o', retriever=None, chain=None):
        self.model_name = model_name
        self.retriever = retriever
        self.chain = chain
        self.context = None
        self.answer = None
        self.docs = None
        self.annotation_chain = None
        self.citations_json = None
        

    def set_model(self, model_name):
        self.model_name = model_name

    def set_retriever(self, retriever):
        self.retriever = retriever

    def set_chain(self, chain):
        self.chain = chain

    def set_annotation_chain(self, chain):
        self.annotation_chain = chain

    # Get relevant documents from the retriever - chunks to be sent to the LLM
    def get_relevant_documents(self, question):
        # Make a POST request to our retriever endpoint with the question as the payload
        t1 = time.time()
        response = requests.post(f"{RETRIEVER_ENDPOINT}/get_relevant_documents/", json={"text": question})
        t2 = time.time()
        print(f"{t2-t1} time taken for retrieval")
        # Parse the JSON response from the retreiever
        json = response.json()
        #convert the json to Document objects - langchain uses Document objects
        relevant_chunks = dict_to_Document_simple(json)
        # Return the list of relevant Document objects
        return relevant_chunks 



    def get_semantic_annotation_chunks(self, documents):
        documents_dict = Document_to_dict(documents)
        node_parser = SentenceSplitter(chunk_size=300, chunk_overlap=20)
        chunks = []
        chunk_id_counter = 0
        for page in documents_dict:
            page_chunks = node_parser.split_text(page["page_content"])
            current_chunk = {}
            for chunk in page_chunks:
                current_chunk = {
                    "page_content": chunk,
                    "metadata": {
                        "page": page["metadata"].get("page"),
                        "source": page["metadata"].get("source"),
                        "summary": page["metadata"].get("summary"),
                        "chunk_id": chunk_id_counter
                    }
                }
                chunk_id_counter += 1
                chunks.append(current_chunk)
        return dict_to_Document_simple(chunks)

    def get_annotations(self, question, answer):
        annotations = self.annotation_chain.invoke({"question": question, "answer": answer, "chunks": format_docs_with_id(self.retrieved_chunks)})
        # print(format_docs_with_id(self.retrieved_chunks))
        return annotations

    # Stream response from the LLM
    async def stream_response(self, question, folder_name):
        rag_model.citation_status = CitationStatus.NOT_STARTED
        # Retrieve the model instance from the llms dictionary using the "model_name"
        model = llms[self.model_name]
        # Get the relevant documents for the given question
        retrieved_docs = self.get_relevant_documents(question)
        # print(f"Retrieved docs: {retrieved_docs}")
        # Store the retrieved documents in the instance variable self.docs
        if not retrieved_docs:
            print("No documents retrieved!")
            return 
        self.docs = retrieved_docs
        # If no relevant documents are retrieved, return immediately
        # Concatenate the content of all retrieved documents with double newline as separator
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        #logger.info(f"Context: {context}")

        # Store the concatenated context in the instance variable self.context
        self.context = context
        # Construct the full prompt for the model including the question and the retrieved document excerpts
        full_prompt = (
            "You area helpful chatbot that has access to my resume and other documents and gives answers based on the question asked "
            f"\n\nHere are the relevant excerpts retrieved from the documents:\n{context}\n\nQuestion: {question}"
        )

        # Create a list of messages with the full prompt as a HumanMessage
        messages = [HumanMessage(content=full_prompt)]
        

        # Stream the response in chunks using the model's astream method
        final_response = ""
        async for chunk in model.astream(messages):
            if hasattr(chunk, "content"):
                final_response += chunk.content
                yield chunk.content
        
        # Store the final response and folder name for later citation generation
        self.final_response = final_response
        self.folder_name = folder_name
        self.question = question
        self.retrieved_chunks = self.get_semantic_annotation_chunks(retrieved_docs)


# Initialize your models
llms = {
    "gpt-4-turbo": ChatOpenAI(model_name="gpt-4-turbo", streaming=True, temperature=0),
    "gpt-3.5-turbo": ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=0),
    "gpt-4o": ChatOpenAI(model_name="gpt-4o", streaming=True, temperature=0),
    #"pplx-70b-online": ChatPerplexity(temperature=0, model="pplx-70b-online")
}

templates_folder = 'templates'

"""# Folder to store cache files - used to store the FAISS index
cache_folder = 'cache'  

# Path to the FAISS index file
faiss_index_path = os.path.join(cache_folder, 'faiss_index')"""

# Initialize the RAG model
rag_model = RAG(model_name='gpt-4o', retriever=None, chain=None)

# Global variable to track the last uploaded folder name
latest_folder_name = None

#This function will be used to generate recommended questions based on the question
def recommended_questions(full_question, answer):
    # we are using 3.5 here since it is cheaper and we dont need something as powerful as gpt-4 to generate recommended questions
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    #prompt
    recommended_questions_prompt = (
        f"Generate 4 follow up questions based on the previous question: '{full_question}' "
        "Return only 4 questions as a numbered list. "
        "Make sure the questions are relevant. "
        "These will be used by the users to ask follow up questions."
    )
    #invoke the model
    response = llm.invoke(recommended_questions_prompt)
    #split the response by newline
    recommended_questions = response.content.split("\n")
    #return the first 4 questions
    return [q.strip() for q in recommended_questions if q.strip()][:4]


# Function to generate a pre-signed URL
def generate_presigned_url(bucket_name, object_name, expiration=385600):
    """
    Generate a pre-signed URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the pre-signed URL to remain valid (default is 3600 seconds)
    :return: Pre-signed URL as string. If error, returns None.
    """
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={
                                                        'Bucket': bucket_name, 
                                                        'Key': object_name,
                                                        'ResponseContentDisposition': 'inline',
                                                        'ResponseContentType': 'application/pdf'
                                                    },
                                                    ExpiresIn=expiration)
    except Exception as e:
        logger.error(e)
        return None
    return response

# Function to delete all objects in a specific folder in the S3 bucket
def delete_folder_from_s3(folder_name: str):
    try:
        # List all objects in the specified folder
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"data/{folder_name}/")
        if 'Contents' in response:
            # Extract keys of all objects in the folder
            objects_to_delete = [{'Key': item['Key']} for item in response['Contents']]
            # Delete objects in batches
            s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': objects_to_delete})
        else:
            logger.info(f"No objects found in folder '{folder_name}'")
    except Exception as e:
        logger.error(f"Error deleting folder '{folder_name}': {str(e)}")
        raise Exception(f"Failed to delete folder '{folder_name}'")

# Function to delete a specific file in a specific folder in the S3 bucket
def delete_file_from_s3(folder_name: str, file_name: str):
    try:
        # Construct the S3 object key
        s3_key = f"data/{folder_name}/{file_name}"
        # Delete the object from the S3 bucket
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        logger.info(f"File '{file_name}' in folder '{folder_name}' successfully deleted")
    except Exception as e:
        logger.error(f"Error deleting file '{file_name}' in folder '{folder_name}': {str(e)}")
        raise Exception(f"Failed to delete file '{file_name}' in folder '{folder_name}'")


# Function to get all PDF files from our cogniq S3 bucket
def get_all_pdfs_from_s3():
    # List all objects in the bucket with the 'data/' prefix
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="data/")
    # Filter out the keys that end with '.pdf' and create a list of PDF files
    pdf_files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.pdf')]
    # Log the list of PDF files found
    logger.info(pdf_files)
    # Return the list of PDF files
    return pdf_files
    
# Function to get PDFs from a specific folder in the S3 bucket
def get_pdfs_from_s3_folder(folder_name: str):
    try:
        # List all objects in the specified folder
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"data/{folder_name}/")
        # Filter and return the keys of the PDF files
        return [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.pdf')]
    except Exception as e:
        logger.error(f"Error listing PDF files in folder '{folder_name}': {str(e)}")
        raise Exception(f"Failed to list PDF files in folder '{folder_name}'")

# Function to setup the chain API with the PDF files
def setup_chain_API(pdf_files):
    # If no PDF files are found, log an error and return
    print('calling setup chain for')
    print(pdf_files)
    setup_annotation_chain()
    if not pdf_files:
        logger.error("No PDF files found to set up the chain.")
        return {"error": "No PDF files found"}

    # Make a POST request to the chain API to setup the chain with the PDF files
    send_file_endpoint = f"{RETRIEVER_ENDPOINT}/send_files"
    # Payload to send the PDF files to the chain API
    payload = {
        "pdf_files": pdf_files
        #print(pdf_files)
    }
    # Make the POST request
    response = requests.post(send_file_endpoint, json=payload)
    # If the response status code is not 200, log an error and return
    if response.status_code != 200:
        return {"error": "Failed to setup chain with the files"}
    # Return the JSON response if the request was successful
    return response.json()

logger.info("calling setup annotation chain")
setup_annotation_chain()

# Define the root route
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# Define the /chat route which deals with streaming responses
@app.post('/chat')
async def test(request: Request):
    global latest_folder_name
    # Parse the JSON body (what we get from the user) of the request asynchronously
    body = await request.json()
    print(body)
    # Validate that the 'messages' key exists and it is not empty
    if "messages" not in body or not body["messages"]:
        raise HTTPException(status_code=400, detail="Messages are required")

    # Extract the content of the last message in the 'messages' list
    question = body["messages"][-1].get("content")
    try:
        folder_name = body["folder"]
        print(f"Folder Name: {folder_name}")
    except:
        folder_name = "data"
        print("Folder Name not provided")

    print(f"Question: {question}")

    # Validate that the extracted content is not empty
    if not question:
        raise HTTPException(status_code=400, detail="Question content is required")
    
    # Check if the folder name has changed
    if folder_name and folder_name != latest_folder_name:
        # Get all PDF files from the specified folder
        pdf_files = get_pdfs_from_s3_folder(folder_name)
        # Setup the chain API with the PDF files from the specified folder
        setup_chain_API(pdf_files)
        # Update the last used folder name
        latest_folder_name = folder_name

    # Define an asynchronous generator function to generate the response in chunks and stream the response
    async def generate():
        response_generator = rag_model.stream_response(question, folder_name)
        async for chunk in response_generator:
            yield f"{chunk}"

    # Return a streaming response using Server-Sent Events (SSE) format
    return StreamingResponse(generate(), media_type='text/event-stream')

# Define the /upload route to get uploaded pdfs and setup the chain
@app.post('/upload')
async def upload_files(
    folder_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    global latest_folder_name

    # Check if any files were uploaded
    if files:
        # Iterate through each file in the uploaded files list
        for file in files:
            try:
                # Asynchronously read the content of the file
                file_content = await file.read()

                # Construct the S3 object key with the user-defined directory and the original file name
                s3_key = f"data/{folder_name}/{file.filename}"

                # Upload the file content to an S3 bucket
                s3_client.put_object(
                    Bucket=BUCKET_NAME,
                    Key=s3_key,
                    Body=file_content,
                    ContentType=file.content_type
                )
            except Exception as e:
                # If there is an error in uploading the file, raise an HTTPException
                raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename} to S3: {str(e)}")

    # After uploading files, call the s3 bucket and get all the files in the specified folder and send them to the chain
    # Get all PDF files from the specified folder in the S3 bucket
    pdf_files = get_pdfs_from_s3_folder(folder_name)
    # Setup the chain API with the PDF files from the specified folder
    setup_chain_API(pdf_files)
    
    # Update the last uploaded folder name
    latest_folder_name = folder_name

    # Return a success message after files are uploaded and processed
    return {"message": "Files successfully uploaded and processed"}


# Function to list PDFs with their pre-signed URLs
@app.get("/getAllFiles", response_model=List[dict])
def list_pdfs():
    try:
        # Get all PDF files from the S3 bucket
        pdf_files = get_all_pdfs_from_s3()
        # Create a list of dictionaries with the file name and pre-signed URL
        pdf_list = [{"fileName": file.split("/")[-1], "url": generate_presigned_url(BUCKET_NAME, file)} for file in pdf_files]
        # Return the list of PDF files with their pre-signed URLs
        return pdf_list
    except Exception as e:
        logger.error(f"Error listing PDF files: {str(e)}")
        raise Exception("Failed to list PDF files")


@app.post('/createFolder')
async def create_folder(request: Request):
    try:
        # Log the entire request form data for debugging
        body = await request.json()
        logger.info(f"Form data: {body.get('folder_name')}")

        # Extract the folder_name from the form data
        folder_name = body.get('folder_name')
        
        # Print the received folder_name
        print(f"Received folder_name: {folder_name}")
        logger.info(f"Received folder_name: {folder_name}")
        
        # Create a placeholder object to represent the folder in S3
        s3_client.put_object(Bucket=BUCKET_NAME, Key=f"data/{folder_name}/")
        return {"message": f"Folder '{folder_name}' created successfully"}
    except Exception as e:
        logger.error(f"Error creating folder: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create folder")

@app.post('/deleteFolder')
async def delete_folder(request: Request):
    try:
        # Parse the JSON body of the request asynchronously
        body = await request.json()
        logger.info(f"Form data: {body.get('folder_name')}")

        # Extract the folder_name from the form data
        folder_name = body.get('folder_name')

        if not folder_name:
            raise HTTPException(status_code=400, detail="Folder name is required")

        # Delete all objects in the specified folder in the S3 bucket
        delete_folder_from_s3(folder_name)

        # Return a success message after the folder is deleted
        return {"message": f"Folder '{folder_name}' successfully deleted"}
    except Exception as e:
        logger.error(f"Error in deleting folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete folder '{folder_name}'")


@app.get('/getFolders')
async def list_folders():
    try:
        # List all objects in the bucket with the 'data/' prefix and group by prefix (folder)
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="data/", Delimiter="/")
        folders = []

        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                # Extract folder name
                folder = prefix['Prefix'].split('/')[-2]
                folders.append(folder)

        # Return the list of folders as a JSON array
        #print(f"Folders found: {folders}")
        logger.info(f"Folders found: {folders}")
        
        return JSONResponse(content=folders)
    except Exception as e:
        logger.error(f"Error listing folders: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list folders")

@app.post('/getFiles')
async def list_files_in_folder(request: Request):
    try:
        # Log the entire request form data for debugging
        body = await request.json()
        logger.info(f"Form data: {body.get('folder_name')}")

        # Extract the folder_name from the form data
        folder_name = body.get('folder_name')

        # Get all PDF files from the specified folder in the S3 bucket
        files = get_pdfs_from_s3_folder(folder_name)
        
        # Create a list of dictionaries with the file name and pre-signed URL
        file_list = [{"fileName": file.split("/")[-1], "url": generate_presigned_url(BUCKET_NAME, file)} for file in files]

        # Return the list of files with their pre-signed URLs as a JSON array
        logger.info(f"Files found in folder '{folder_name}': {file_list}")
        return JSONResponse(content=file_list)
    except Exception as e:
        logger.error(f"Error listing files in folder: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list files in folder")

@app.post('/deleteFile')
async def delete_file(request: Request):
    try:
        # Parse the JSON body of the request asynchronously
        body = await request.json()
        logger.info(f"Form data: {body}")

        # Extract the folder_name and file_name from the form data
        folder_name = body.get('folder_name')
        file_name = body.get('file_name')

        if not folder_name or not file_name:
            raise HTTPException(status_code=400, detail="Folder name and file name are required")

        # Delete the specified file in the specified folder in the S3 bucket
        delete_file_from_s3(folder_name, file_name)

        # Return a success message after the file is deleted
        return {"message": f"File '{file_name}' in folder '{folder_name}' successfully deleted"}

        # Get all PDF files from the specified folder in the S3 bucket
        pdf_files = get_pdfs_from_s3_folder(folder_name)
        # Setup the chain API with the PDF files from the specified folder
        setup_chain_API(pdf_files)

    except Exception as e:
        logger.error(f"Error in deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file '{file_name}' in folder '{folder_name}'")
if _name_ == '_main_':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
