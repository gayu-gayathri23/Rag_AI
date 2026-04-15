# Rag_AI
### RAG algorithm using Hugging Face Embedding 

RAG is a technique that enhances the capibilities of LLMs by 
combining the infomration retrieval with the text generation

Instead of replying on pre-trained knowledge , RAG fetch the 
relevant data from external source and us it to generate more accurate response.

Packages

streamlit 
python-dotenv
google-generativeai
PyPDF2

langchain  # core frameworks
langchain-community  # connect huggingface models to perform embedding
faiss-cpu # Fast vector database to store the embedded data 
langchain-huggingface # connect huggingface models to perform embedding
langchain-text-splitters #to split the data into chunks
sentence-tranformers # pre-trained models to convert chunks into vectors
langchain-core # to handle documents, chains of data etc.,.
