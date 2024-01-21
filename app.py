import os, tempfile
import streamlit as st
from io import StringIO
from llama_index.llms import OpenAI, Gemini, Cohere
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceSplitter

def upload_file():
    file = st.file_uploader("Upload a file")
    if file is not None:
        file_path = save_uploaded_file(file)
        
        if file_path:
            # Use the file with SimpleDirectoryReader
            # Replace 'SimpleDirectoryReader' with the actual code to read the file
            # For example:
            # reader = SimpleDirectoryReader(input_files=[file_path])
            loaded_file = SimpleDirectoryReader(input_files=[file_path]).load_data()
            print(f"Total documents: {len(loaded_file)}")

            # If there are any documents, print the details of the first one
            if loaded_file:
                print(f"First document, id: {loaded_file[0].doc_id}")
                #print(f"First document, hash: {loaded_file[0].hash}")
                #print(f"First document, text ({len(loaded_file[0].text)} characters):\n{'='*20}\n{loaded_file[0].text[:360]} ...")

            st.success(f"File uploaded successfully")
            #print(loaded_file)
        return loaded_file
    return None

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Placeholder functions for different components of the RAG pipeline
# These functions should be replaced with actual implementations


def select_llm():
    st.header("Choose LLM")
    llm_choice = st.selectbox("Select LLM", ["GPT-3.5", "GPT-4", "Gemini", "Cohere"])
    
    if llm_choice == "GPT-3.5":
        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")
        #print(llm.complete("which specific version of gpt model are you. what is the model name in api? "))
    elif llm_choice == "GPT-4":
        llm = OpenAI(temperature=0.1, model="gpt-4-1106-preview")
        #print(llm.complete("which specific version of gpt model are you. what is the model name in api? "))
    elif llm_choice == "Gemini":
        llm = Gemini(model="models/gemini-pro")
        print(llm.complete("Which llm are you? "))
    elif llm_choice == "Cohere":
        llm = Cohere(model="command", api_key=st.secrets['COHERE_API_TOKEN'])
        print(llm.complete("Which llm are you? "))
    return llm

def select_embedding_model():
    st.header("Choose Embedding Model")
    model_names = [
        "WhereIsAI/UAE-Large-V1",
        "BAAI/bge-large-en-v1.5",
        "khoa-klaytn/bge-small-en-v1.5-angle",
        "BAAI/bge-base-en-v1.5",
        "llmrails/ember-v1",
        "jamesgpt1/sf_model_e5",
        "thenlper/gte-large",
        "infgrad/stella-base-en-v2",
        "thenlper/gte-base",
        "BAAI/bge-small-en-v1.5"
    ]
    selected_model = st.selectbox("Select Embedding Model", model_names)
    embed_model = HuggingFaceEmbedding(model_name=selected_model)

    return embed_model

def select_node_parser(file):
    node_parser = st.selectbox("Choose Chunking Method", ["Method1", "Method2", "Method3"])
    
    chunk_size=100
    chunk_overlap=5

    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Use the parser to get nodes from documents
    nodes = parser.get_nodes_from_documents(file)

        # Initialize the language model
    llm = OpenAI(model_name="gpt-3.5-turbo")

    # Create the service context
    service_context = ServiceContext.from_defaults(llm=llm, node_parser=parser)

    # Create the vector index
    vector_index = VectorStoreIndex.from_documents(documents=file, service_context=service_context, show_progress=True)
    # Print the number of nodes in the index
    print(f"Total nodes: {vector_index._get_node_with_embedding}")

    return node_parser

def select_retrieval_method():
    retrieval_method = st.selectbox("Choose Retrieval Method", ["Method1", "Method2", "Method3"])
    return retrieval_method

def select_response_synthesis_method():
    response_synthesis_method = st.selectbox("Choose Response Synthesis Method", ["Method1", "Method2", "Method3"])
    return response_synthesis_method

def select_vector_store():
    vector_store = st.selectbox("Choose Vector Store", ["Store1", "Store2", "Store3"])
    return vector_store

def select_query_engine():
    query_engine = st.selectbox("Choose Query Engine", ["Engine1", "Engine2", "Engine3"])
    return query_engine

def generate_rag_pipeline():
    st.session_state['pipeline_generated'] = True
    st.success("RAG Pipeline Generated Successfully!")

def send_query():
    query = st.session_state['query']
    # Placeholder for sending query to LLM and getting the response
    # Replace this with actual query handling and response generation
    response = f"Response for the query: {query}"
    st.markdown(response)

def main():
    st.title("RAG Tester Application")

    # Upload file
    file = upload_file()

    # Select RAG components
    llm = select_llm()
    embed_model = select_embedding_model()
    #embeddings = embed_model.get_text_embedding("Hello World!")
    #print(len(embeddings))
    #print(embeddings[:5])

    node_parser = select_node_parser(file)
    retrieval_method = select_retrieval_method()
    response_synthesis_method = select_response_synthesis_method()
    vector_store = select_vector_store()
    query_engine = select_query_engine()

    # Generate RAG Pipeline Button
    if st.button("Generate RAG Pipeline"):
        generate_rag_pipeline()

    # After generating the RAG pipeline
    if st.session_state.get('pipeline_generated', False):
        query = st.text_input("Enter your query", key='query')
        if st.button("Send"):
            send_query()

if __name__ == "__main__":
    main()
