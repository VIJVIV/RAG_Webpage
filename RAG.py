import os
from getpass import getpass
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

load_dotenv()
HF_token = os.getenv('HUGGINGFACEHUB_API_TOKEN') 

#Load data from webpage url
def load_data(url):
    data = WebBaseLoader(url)
    content = data.load()
    return content

#Text splitter - creating 'chunks of data'
def text_splitter(content):  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50) 
    chunks = text_splitter.split_documents(content)
    print('Number of chunks created - ', len(chunks))   #Chunk size is number of characters, not tokens
    for i, _ in enumerate(chunks):
        print(f'chunk # {i}, {chunks[i]}')
        print('----------------------------------------')
    return chunks

#Embedding generation
def embedding(HF_token): 
    embedding_model = 'BAAI/bge-base-en-v1.5' #Refer MTEB HuggingFace leaderboard for choosing embedding model
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key = HF_token, model_name = embedding_model)
    return embeddings

#Vector db creation
def vector_database(chunks, embeddings):
    client = chromadb.Client()
    vector_db = Chroma.from_documents(chunks, embeddings)
    print('vector db created with embeddings')
    return vector_db

#RAG implementation
def RAG(vector_db, user_query): 
    model = HuggingFaceHub(repo_id='HuggingFaceH4/zephyr-7b-alpha', model_kwargs={'temperature': 0.5, 
                                                                                  'max_new_tokens': 512,
                                                                                  'max_length': 64}) #LLM Model

    #Step 1 - Retrieval from vector db
    retriever = vector_db.as_retriever(search_type='mmr', search_kwargs={'k':1})   #Retrieves k relevant chunks
    #Tune k value according to content, for complex data such as pdf/collection_of_pdf's, k can be 5-10

    #Step 2 - Augment - Prompt template
    query = user_query
    prompt = f'''
    <|system|>
    You are an AI assistant that follows instructions extremely well. Please be truthful and give direct answers
    </s>
    <|user|>
    {query}
    </s>
    <|chatbot|>
    '''

    #Step 3 - Generation - Connect LLM with retrieved document chunk
    qa = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type='refine')
    response = qa(prompt)
    return response['result'].split("<|>")[-1].strip()


def get_response(url, user_query):
    content = load_data(url)

    chunking = text_splitter(content)
    embeddings = embedding(HF_token)
    vector_db = vector_database(chunking, embeddings)

    response = RAG(vector_db, user_query)
    return(response)
