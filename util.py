import os
import openai
import numpy as np
from PyPDF2 import PdfReader
from typing import List,Tuple,Optional
from langchain_core.documents import Document
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback
from pdf2image import convert_from_path
import spacy


def spacy_sim(str1, str2)-> (float):
    """Give similarity score using spacy.load("en_core_web_md")

    Args:
        str1 (str): string you want to compare
        str2 (str): string you want to compare to
        

    Returns:
        float: similarity score
    """
    nlp = spacy.load("en_core_web_md")
    doc_1 = nlp(str1)
    doc_2 = nlp(str2)
    return doc_1.similarity(doc_2)

def openai_sim(str1, str2)-> (float):
    """Give similarity score using OpenAi "text-embedding-ada-002"

    Args:
        str1 (str): string you want to compare
        str2 (str): string you want to compare to
        

    Returns:
        float: similarity score
    """
    # Call the API
    client = openai.Client()  # Create a client instance

    # Call the API using the new structure
    response = client.embeddings.create(
        input=[str1, str2],
        model="text-embedding-ada-002"  # Replace with the actual latest model
    )

    # Extract the embeddings
    embedding1 = response.data[0].embedding  # Use dot notation for accessing attributes
    embedding2 = response.data[1].embedding

    # Calculate cosine similarity
    similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity_score
@st.cache_resource
def cached_get_text_chunks_from_pdf(_loaded_data, *args, **kwargs):
    return _get_text_chunks_from_pdf(_loaded_data, *args, **kwargs)
def _get_text_chunks_from_pdf(loaded_data:PdfReader)-> (Tuple[List[str],dict]):
    """Extract /Split and Create a dict that contain add the text of the DOC 
    OP_DICT_STRUCTURE = {page content: page No}

    Args:
        loaded_data (PdfReader): _description_

    Returns:
        _type_: _description_
    """
    print("get_text_chunks_from_pdf executed")
    text = ""
    page_dict = {}
    for i, page in enumerate(loaded_data.pages):
        page_content = page.extract_text()
        text += page_content + '\n\n'
        page_dict[page_content] = i+1
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks,page_dict

def _create_similarity_search_docs(_chunks:List[str], question:str,OpenAI_SECRET_KEY:str) ->(List[Document]):
    """Create a FAISS similarity_search across the whole doc and return similarity_search 

    Args:
        _chunks (List[str]): list of text string in the DOC
        question (str):user Query about the DOC
        OpenAI_SECRET_KEY (str): your OpenAI_SECRET_KEY
    Returns:
        similarity_search_docs (List[Document]): List of similar Document matching the user query
    """
    print("create_knowledge_base executed")
    embeddings = OpenAIEmbeddings(openai_api_key=OpenAI_SECRET_KEY)
    faiss_index = FAISS.from_texts(_chunks, embeddings)
    similarity_search_docs = faiss_index.similarity_search(question)
    return similarity_search_docs

def get_llm_response(similarity_search_docs:List[Document], question:str,page_dict: dict,OpenAI_SECRET_KEY: str,use_similarity:str = "spacy_sim")-> (Tuple[str,List[list]]):
    """
    Get the response fom the LLM and also crete a dict containing list of pages with most similarity to response

    Args:
        similarity_search_docs (List[Document]): List of faiss_index Documents created by FAISS.from_texts(_chunks, embeddings).similarity_search
        question (str): Query asked by user
        page_dict (dict): Dict of the PDF {Page Content:Page Number}
        OpenAI_SECRET_KEY (str): your OpenAI_SECRET_KEY
        use_similarity (str) : default = "spacy_sim" "spacy_sim" | 'openAI_sim' 

    Returns:
        response (str): response from OpenAI of the Doc
        data (List[list]): returns the list of pages with most similarity to response
    """
    llm = OpenAI(openai_api_key=OpenAI_SECRET_KEY)
    chain = load_qa_chain(llm)
    with get_openai_callback() as cb:
        response = chain.run(input_documents=similarity_search_docs,
                                question=question)
        print(f'billing details: {cb}')

    print("response = ",response)
    data = []
    if use_similarity == "spacy_sim":
        for page_content, page_num in page_dict.items():
            similarity = spacy_sim(response, page_content)
            data.append([similarity, page_num])
    elif use_similarity == "openAI_sim":
        for page_content, page_num in page_dict.items():
            similarity = openai_sim(response, page_content)
            data.append([similarity, page_num])
    else:
        raise ValueError("Invalid Value for use_similarity param")
    # Sort the similarity score from high to low.
    data = sorted(data, key=lambda x: x[0], reverse=True)
    return response,data

def get_answers_from_pdf(pdf,user_question,OpenAI_SECRET_KEY):
    st.session_state["submit"] == True
    loader = PdfReader(pdf)
    
    chunks,page_dict = cached_get_text_chunks_from_pdf(loader)
    docs = _create_similarity_search_docs(chunks,user_question,OpenAI_SECRET_KEY)
    response,data = get_llm_response(docs,user_question,page_dict,OpenAI_SECRET_KEY)
    st.write(f"Answer: {response}")
    images = convert_from_path(os.path.join("./store_pdf",pdf.name))
    
    # Generate images per page from the pdf.
    for page_details in data:
        page_number = page_details[1]-1
        similarity_score = page_details[0]
        st.write(f"page_number = {page_number} similarity Score = {similarity_score}")
        st.image(images[page_number])
        
    

    # Show the page image with the highest similarity.
    