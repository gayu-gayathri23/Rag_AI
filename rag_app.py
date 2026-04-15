import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings 
# to get embedding model 
from langchain_core.documents import Document #to store text and metdata
from langchain_text_splitters import CharacterTextSplitter # to split large paragraphs into chunks
from langchain_community.vectorstores import FAISS # to store the embedding data from given document for similarity search 


key=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

model=genai.GenerativeModel('gemini-2.5-flash-lite')


def load_embedding():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

with st.spinner('Loading embedding model...⏳'):
    embedding_model= load_embedding()
st.set_page_config('RAG DEMO',page_icon='🎯',layout='wide')
st.title('RAG Assistant :blue[Using Embedding and LLM] 📚💻✍🏼📓')
st.subheader(':green[Your Intelligent Document Assistant 👩🏻‍💻]')

uploaded_file=st.file_uploader('Upload your document here 📝',type=['pdf','txt'])

if uploaded_file:
    pdf=PdfReader(uploaded_file)
    raw_text=''
    for page in pdf.pages:
        raw_text+=page.extract_text()
        
    if raw_text.strip(): # remove spaces and checkw whether have text data
        #and ensures that given raw_text is not empty
        
        doc=Document(page_content=raw_text)
        # to get content in the given pdf and metadata 
        
        splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        
        # max char in each chunk is 1000 and overlap to maintain the relation btw context=200
        
        chucnk_text=splitter.split_documents([doc])
        # to split the given document into smaller chunks for better processing and understanding
        
        text=[i.page_content for i in chucnk_text]
        # to get data as list of smaller text 
        
        vector_db=FAISS.from_texts(text,embedding_model)
        
        retrive=vector_db.as_retriever()
        #create a search tool to find relevant chunks
        
        st.success('Document uploaded and processed successfully!! ✅ Ask a question now 🤔')
                  
        query=st.text_input('Ask a question about your document here 💬')
        
        if query:
            with st.chat_message('human 🙋🏻‍♀️'):
                with st.spinner('Analyzing the document....🧐'):
                    relevant_data=retrive.invoke(query)
                    #invoke the embedding modela and search the similar chunk in FAISS for the given query
                    
                    content='\n\n'.join([ i.page_content for i in relevant_data])
                    
                    prompt=f'''
                    You are an AI expert. Use the generated content {content}
                    to answer the question: {query}. If you are not sure with the answer, say 
                    "I have no content related to this question. Please ask
                    relevant query to answer"
                    
                    Result in bullet points'''
                    
                    response=model.generate_content(prompt)
                    st.markdown('## :blue[Results 🔍]')
                    st.write(response.text)
                    
    else:
        st.warning('Drop the file in PDF format ')
                    
                    

