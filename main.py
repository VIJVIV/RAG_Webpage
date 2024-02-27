import os
from getpass import getpass
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

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
def embedding(HF_token, embedding_model): 
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key = HF_token, model_name = embedding_model)
    return embeddings

#Vector db creation
def vector_database(chunks, embeddings):
    vector_db = Chroma.from_documents(chunks, embeddings)
    print('vector db created with embeddings')
    return vector_db


if __name__ == '__main__':
    HF_token = getpass()
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_token

    URL = 'https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/'

    embedding_model_name = 'BAAI/bge-base-en-v1.5' #Refer MTEB HuggingFace leaderboard for choosing embedding model

    #Read and load data from specified webpage
    data = WebBaseLoader(URL)
    content = data.load()

    #Chunking, embedding and vector database creation
    chunking = text_splitter(content)
    embeddings = embedding(HF_token, embedding_model_name)
    vector_db = vector_database(chunking, embeddings)

    #LLM Model 
    model = HuggingFaceHub(repo_id='HuggingFaceH4/zephyr-7b-alpha', model_kwargs={'temperature': 0.5, 
                                                                                  'max_new_tokens': 512,
                                                                                  'max_length': 64})
    
    #Step 1 - Retrieval from vector db
    retriever = vector_db.as_retriever(search_type='mmr', search_kwargs={'k':1})   #Retrieves k relevant chunks
    #Tune k value according to content, for complex data such as pdf/collection_of_pdf's, k can be 5-10

    #Step 2 - Augment - Prompt template
    query = "Did NVIDIA develop an AI workflow for retrieval augmented generation? What does it include?"

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
    print(response['result'])

