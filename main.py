import os
from getpass import getpass
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

HF_token = getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_token

URL = 'https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/'

#Read and load data from specified webpage
data = WebBaseLoader(URL)
content = data.load()
print(content)

#Text splitter - creating 'chunks of data'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)  
chunking = text_splitter.split_documents(content)
print(len(chunking))

#Embedding model - Refer to MTEB HuggingFace leaderboard for choosing appropriate embedding model 
embeddings = HuggingFaceInferenceAPIEmbeddings(api_key = HF_token, model_name = 'BAAI/bge-base-en-v1.5')

#Vector db creation
vector_db = Chroma.from_documents(chunking, embeddings)
print('Done')

#Model 
model = HuggingFaceHub(repo_id='HuggingFaceH4/zephyr-7b-alpha', model_kwargs={'temperature': 0.5, 
                                                                              'max_new_tokens': 512,
                                                                              'max_length': 64})

#Step 1 - Retrieval from vector db
retriever = vector_db.as_retriever(search_type='mmr', search_kwargs={'k':1})   #Retrieves k relevant chunks
#Tune k value according to content, for complex data such as pdf/collection of pdf's, k can be 5-10

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
qa = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type='stuff')
response = qa(prompt)
print(response['result'])

