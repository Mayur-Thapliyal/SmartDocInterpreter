import streamlit as st
import base64
from util import *
from os.path import exists


st.set_page_config(page_title="pdf-Insight", page_icon="📖", layout="wide")
st.header("pdf-Insight")
col1, col2 = st.columns(spec=[2, 1], gap="small")

def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")
        
        
def clear_submit():
    st.session_state["submit"] = False
    
def download_file(uploaded_file):
    file_exists = exists(os.path.join("./store_pdf",uploaded_file.name))
    if not file_exists:
        with open(os.path.join("./",uploaded_file.name),"wb") as f: 
            f.write(uploaded_file.getbuffer())  
            return True
    return False

def displayPDF(uploaded_file):
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')

    # Embed PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:
    OpenAI_SECRET_KEY = st.text_input(label='OpenAI_SECRET_KEY',type="password",placeholder="Enter your OpenAI key here",value=None,)
    uploaded_file = st.file_uploader(
        "Upload file", type=["pdf"], 
        help="Only PDF files are supported", 
        on_change=clear_submit)
    if uploaded_file:
        is_file_downloaded = download_file(uploaded_file)
        print("file downloaded", is_file_downloaded)

if uploaded_file:
    
    with col1:
        displayPDF(uploaded_file)
    
    with col2:
        question = st.text_input(
            "Ask something about the article",
            placeholder="Can you give me a short summary?",
            disabled=not uploaded_file,
            on_change=clear_submit,
        )
        print(type(OpenAI_SECRET_KEY),"<<<<<<<<<<<<<<<<<<<<<<<<<")
        if OpenAI_SECRET_KEY == None or OpenAI_SECRET_KEY == "":
            st.info(f"Enter OpenAI_SECRET_KEY ")
        if uploaded_file and question and st.session_state["submit"] == False and OpenAI_SECRET_KEY:
            st.info("Answer")
            try:
                get_answers_from_pdf(uploaded_file,question,OpenAI_SECRET_KEY)
                delete_files_in_directory("./store_pdf")
            except Exception as e:
                st.write(f"debug message = {str(e)}")
            
            print("Done")
