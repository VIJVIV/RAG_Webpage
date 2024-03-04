import streamlit as st
import RAG

def process_RAG(webpage, user_query):
    answer = RAG.get_response(webpage, user_query)
    st.write(answer)

def main():
    st.set_page_config(page_title='Query a Website App')
    st.title('Query a website 🦜🔗')

    webpage = st.text_input('Enter a website link')
    
    user_query = st.text_input('Please ask your query for the webpage')
    st.button('Submit')

    HF_token = st.secrets['HUGGINGFACEHUB_API_TOKEN']

    if webpage and user_query:
        with st.spinner('Processing...'):
            process_RAG(webpage, user_query, HF_token)


if __name__ == '__main__':
    main()