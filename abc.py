import streamlit as st
import PyPDF2
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Initialize RAG components (pre-trained RAG model)
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

# Function to answer a query based on uploaded document
def answer_query(query, document_text):
    inputs = tokenizer(query, return_tensors="pt")
    generated = model.generate(
        input_ids=inputs["input_ids"],
        retriever_input_ids=tokenizer(document_text, return_tensors="pt")["input_ids"]
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Streamlit UI
st.title("RAG Document Query System")

# Upload a document (PDF)
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# Extract text from the uploaded PDF
if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    st.write("Document Uploaded Successfully!")
    st.write("Document Preview:")
    st.write(text[:500])  # Show a snippet of the document

# Input box for user to ask a question
query = st.text_input("Ask a question based on the document:")

# Process the query and return an answer
if st.button("Get Answer"):
    if uploaded_file and query:
        answer = answer_query(query, text)
        st.write("Answer:")
        st.write(answer)
    else:
        st.write("Please upload a document and enter a query.")
