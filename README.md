# Retrieval Augmented Generation (RAG) model for webpage query

This repository showcases the implementation of a basic RAG model to query a webpage. The RAG model is implemented purely using open source models, leveraging the langchain framework. You would require an api token for accessing the hugging face models, which would need to be inserted as a password during code execution. This project is a learning exercise, expected to evolve or lead to a more comprehensive project in the (near) future.


Upon code execution, the following image shows the context (i.e. chunk retrieved from vector db as relevant context for the llm) and the response generated thereby:

<img width="878" alt="RAG code output" src="https://github.com/VIJVIV/RAG_Webpage/assets/146338220/7bca814b-7a9e-4d4b-ad36-6c45cff46718">






To further confirm that the model is not halucinating, the webpage content/context for the respective query is:

<img width="878" alt="Nvidia webpage" src="https://github.com/VIJVIV/RAG_Webpage/assets/146338220/825424a0-a363-4040-a030-fc6e4312c9c3">

