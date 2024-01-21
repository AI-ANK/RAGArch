import os, tempfile, faiss, qdrant_client
import streamlit as st
from io import StringIO
from llama_index.llms import OpenAI, Gemini, Cohere
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceSplitter, CodeSplitter, SemanticSplitterNodeParser, TokenTextSplitter
from llama_index.node_parser.file import HTMLNodeParser, JSONNodeParser, MarkdownNodeParser
from llama_index.vector_stores import FaissVectorStore, MilvusVectorStore, QdrantVectorStore

@st.cache_data(experimental_allow_widgets=True) 
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

@st.cache_data
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
    llm_choice = st.selectbox("Select LLM", ["Gemini", "Cohere", "GPT-3.5", "GPT-4"])
    
    if llm_choice == "GPT-3.5":
        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")
        st.write(f"{llm_choice} selected")
        #print(llm.complete("which specific version of gpt model are you. what is the model name in api? "))
    elif llm_choice == "GPT-4":
        llm = OpenAI(temperature=0.1, model="gpt-4-1106-preview")
        st.write(f"{llm_choice} selected")
        #print(llm.complete("which specific version of gpt model are you. what is the model name in api? "))
    elif llm_choice == "Gemini":
        llm = Gemini(model="models/gemini-pro")
        st.write(f"{llm_choice} selected")
    elif llm_choice == "Cohere":
        llm = Cohere(model="command", api_key=st.secrets['COHERE_API_TOKEN'])
        st.write(f"{llm_choice} selected")
    return llm

def select_embedding_model():
    st.header("Choose Embedding Model")
    model_names = [
        "BAAI/bge-small-en-v1.5",
        "WhereIsAI/UAE-Large-V1",
        "BAAI/bge-large-en-v1.5",
        "khoa-klaytn/bge-small-en-v1.5-angle",
        "BAAI/bge-base-en-v1.5",
        "llmrails/ember-v1",
        "jamesgpt1/sf_model_e5",
        "thenlper/gte-large",
        "infgrad/stella-base-en-v2",
        "thenlper/gte-base"
    ]
    with st.spinner("Please wait"):
        selected_model = st.selectbox("Select Embedding Model", model_names)
        embed_model = HuggingFaceEmbedding(model_name=selected_model)
        st.session_state['embed_model'] = embed_model
        st.write(F"Embedding model selected: {embed_model}")

    return embed_model

def select_node_parser():
    st.header("Choose Node Parser")
    parser_type = st.selectbox("Select Node Parser", ["SentenceSplitter", "CodeSplitter", "SemanticSplitterNodeParser", 
                                                     "TokenTextSplitter", "HTMLNodeParser", "JSONNodeParser", "MarkdownNodeParser"])

    parser = None
    if parser_type == "HTMLNodeParser":
        tags = st.text_input("Enter tags separated by commas", "p, h1")
        tag_list = tags.split(',')
        parser = HTMLNodeParser(tags=tag_list)

    elif parser_type == "JSONNodeParser":
        parser = JSONNodeParser()

    elif parser_type == "MarkdownNodeParser":
        parser = MarkdownNodeParser()

    elif parser_type == "CodeSplitter":
        language = st.text_input("Language", "python")
        chunk_lines = st.number_input("Chunk Lines", min_value=1, value=40)
        chunk_lines_overlap = st.number_input("Chunk Lines Overlap", min_value=0, value=15)
        max_chars = st.number_input("Max Chars", min_value=1, value=1500)
        parser = CodeSplitter(language=language, chunk_lines=chunk_lines, chunk_lines_overlap=chunk_lines_overlap, max_chars=max_chars)

    elif parser_type == "SentenceSplitter":
        chunk_size = st.number_input("Chunk Size", min_value=1, value=1024)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, value=20)
        parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif parser_type == "SemanticSplitterNodeParser":
        if 'embed_model' not in st.session_state:
            st.warning("Please select an embedding model first.")
            return None
    
        embed_model = st.session_state['embed_model']

        buffer_size = st.number_input("Buffer Size", min_value=1, value=1)
        breakpoint_percentile_threshold = st.number_input("Breakpoint Percentile Threshold", min_value=0, max_value=100, value=95)
        # Ensure embed_model is initialized or available here
        parser = SemanticSplitterNodeParser(buffer_size=buffer_size, breakpoint_percentile_threshold=breakpoint_percentile_threshold, embed_model=embed_model)

    elif parser_type == "TokenTextSplitter":
        chunk_size = st.number_input("Chunk Size", min_value=1, value=1024)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, value=20)
        parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)        

    return parser


def select_response_synthesis_method():
    st.header("Choose Response Synthesis Method")
    response_modes = [
        "refine",
        "tree_summarize",  
        "compact", 
        "simple_summarize", 
        "no_text", 
        "accumulate", 
        "compact_accumulate"
    ]
    selected_mode = st.selectbox("Select Response Mode", response_modes)
    response_mode = selected_mode
    return response_mode

def select_vector_store():
    st.header("Choose Vector Store")
    vector_stores = ["Simple", "Faiss", "Milvus", "Qdrant"]
    selected_store = st.selectbox("Select Vector Store", vector_stores)

    vector_store = None
    if selected_store == "Faiss":
        d = 1536
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

    elif selected_store == "Milvus":
        vector_store = MilvusVectorStore(dim=1536, overwrite=True)

    elif selected_store == "Qdrant":
        client = qdrant_client.QdrantClient(location=":memory:")
        vector_store = QdrantVectorStore(client=client, collection_name="sampledata")

    return vector_store

@st.cache_data
def generate_rag_pipeline(file, llm, embed_model, node_parser, response_mode, vector_store):
    if vector_store is not None:
        # Set storage context if vector_store is not None
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    else:
        storage_context = None

    # Create the service context
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)

    # Create the vector index
    vector_index = VectorStoreIndex.from_documents(documents=file, storage_context=storage_context, service_context=service_context, show_progress=True)
    if storage_context:
        vector_index.storage_context.persist(persist_dir="persist_dir")

    # Create the query engine
    query_engine = vector_index.as_query_engine(
        response_mode=response_mode,
        verbose=True,
    )

    return query_engine

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

    node_parser = select_node_parser()
    # Process nodes only if a file has been uploaded
    if file is not None:
        if node_parser:
            nodes = node_parser.get_nodes_from_documents(file)
            print(f"First node: {nodes[0].text}")

    response_mode = select_response_synthesis_method()
    vector_store = select_vector_store()

    # Generate RAG Pipeline Button
    if file is not None:
        if st.button("Generate RAG Pipeline"):
            query_engine = generate_rag_pipeline(file, llm, embed_model, node_parser, response_mode, vector_store)
            st.session_state['query_engine'] = query_engine
            st.session_state['pipeline_generated'] = True
            st.success("RAG Pipeline Generated Successfully!")
    else:
        st.error('Please upload a file')


    # After generating the RAG pipeline
    if st.session_state.get('pipeline_generated', False):
        query = st.text_input("Enter your query", key='query')
        if st.button("Send"):
            if 'query_engine' in st.session_state:
                response = st.session_state['query_engine'].query(query)
                st.markdown(response, unsafe_allow_html=True)
            else:
                st.error("Query engine not initialized. Please generate the RAG pipeline first.")

if __name__ == "__main__":
    main()
