# Retrieval Augmented Generation (RAG) model for webpage query

This repository showcases the implementation of a basic RAG model to query a webpage. The RAG model is implemented purely using open source models, leveraging the langchain framework. You would require an api token for accessing the hugging face models. This project is a learning exercise, expected to evolve (which will be reflected as commits and readme file updates) or lead to a more comprehensive project in the (near) future.


**Upon code execution, the following image shows the system prompt, the query posed by the user, and the response generated by the chatbot thereby:**

<img width="878" alt="RAG code output" src="https://github.com/VIJVIV/RAG_Webpage/assets/146338220/e48171b5-b292-4e00-b005-2fb6d80c2651">


&nbsp;  




**To confirm that the model is not halucinating, the webpage content/context for the respective query is as follows:**

<img width="878" alt="Nvidia webpage" src="https://github.com/VIJVIV/RAG_Webpage/assets/146338220/825424a0-a363-4040-a030-fc6e4312c9c3">


&nbsp;  




**Update 1: Integration with Streamlit to build a webpage query app**



https://github.com/VIJVIV/RAG_Webpage/assets/146338220/7566b2b9-ed38-4490-bc4f-b730a3662be9

&nbsp;


Project dependencies can be installed using the requirements file
```bash
pip install -r requirements.txt
```

Save your Hugging Face api token in a '.env' file before code execution. Run program from terminal using:
```bash
streamlit run main.py
```

