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

from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

bedrock_client = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock_client)

folder_path = '/tmp/'

def get_llm():
    llm = Bedrock(model_id='anthropic.claude-v2:1',
                  client=bedrock_client,
                  model_kwargs={'max_tokens_to_sample':512})
    return llm

# get response using LLM
def get_response(llm, faiss_index, question):
    prompt_template = """
    Human: Use the following pieces of context to provide a concise answer to the question at the end
    but atleast provide answer with 250 words.if you don't know answer you can say don't know.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=['context','question']
    )

    qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=faiss_index.as_retriever(
                search_type='similarity', search_kwargs={'k':5}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': PROMPT}
        )
    response = qa({'query': question})
    return response['result']

#load index
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key='my_faiss.faiss', Filename=f'{folder_path}my_faiss.faiss')
    s3_client.download_file(Bucket=BUCKET_NAME, Key='my_faiss.pkl', Filename=f'{folder_path}my_faiss.pkl')
def main():
    st.header('This is client site for Chat with PDF using Bedrock, RAG etc.')

    load_index()
    dir_list = os.listdir(folder_path)
    st.write(f'Files and Directories in {folder_path}')
    st.write(dir_list)
    # Cretae index 
    faiss_index = FAISS.load_local(index_name='my_faiss,
                                   folder_path=folder_path,
                                   embeddings=bedrock_embeddings
                                   )
    st.write('INDEX IS REDAY!!')
    question = st.text_input('Pleae ask your question')
    if st.button('Ask Questions'):
        with st.spinner('Querying...'):
            llm = get_llm()

            # get response
            response = get_response(llm, faiss_index, question)
            st.write(f'Response: {response}')


if __name__ == '__main__':
    main()