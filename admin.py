import boto3
import streamlit as st
import os
import uuid

# s3 client
s3_client = boto3.client('s3')
BUCKET_NAME = os.getenv('BUCKET_NAME')

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

# import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock_client)

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

# Create Vector Store
def create_vector_store(request_id, documents):
    vector_store_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f'{request_id}.bin'
    folder_path='/tmp/'
    vector_store_faiss.save_local(index_name=file_name, folder_path=folder_path)
    ## upload to S3
    s3_client.upload_file(Filename=folder_path + '/' + file_name + '.faiss', Bucket=BUCKET_NAME, Key='my_faiss.faiss')
    s3_client.upload_file(Filename=folder_path + '/' + file_name + '.pkl', Bucket=BUCKET_NAME, Key='my_faiss.pkl')
    return True

#main method
def main():
    st.write("Admin site for Chat With PDF Demo!")
    uploaded_file = st.file_uploader("Choose a pdf file", type=["pdf"])
    if uploaded_file is not None:
        request_id = str(uuid.uuid4())
        st.write(f'Request ID: {request_id}')
        saved_file_name = f'{request_id}.pdf'
        with open(saved_file_name, 'wb') as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f'Total pages: {len(pages)}')

        # Split Text
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f'Number of splitted documents: {len(splitted_docs)}')
        
        st.write('Creating Vector Store:')
        result = create_vector_store(request_id, splitted_docs)
        if result:
            st.write('PDF created successfully')

        
if __name__ == "__main__":
    main()
