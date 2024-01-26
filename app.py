import os, tempfile, qdrant_client
import streamlit as st
from llama_index.llms import OpenAI, Gemini, Cohere
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceSplitter, CodeSplitter, SemanticSplitterNodeParser, TokenTextSplitter
from llama_index.node_parser.file import HTMLNodeParser, JSONNodeParser, MarkdownNodeParser
from llama_index.vector_stores import QdrantVectorStore, PineconeVectorStore
from pinecone import Pinecone


def reset_pipeline_generated():
    if 'pipeline_generated' in st.session_state:
        st.session_state['pipeline_generated'] = False

def upload_file():
    file = st.file_uploader("Upload a file", on_change=reset_pipeline_generated)
    if file is not None:
        file_path = save_uploaded_file(file)
        
        if file_path:
            loaded_file = SimpleDirectoryReader(input_files=[file_path]).load_data()
            print(f"Total documents: {len(loaded_file)}")

            st.success(f"File uploaded successfully. Total documents loaded: {len(loaded_file)}")
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


def select_llm():
    st.header("Choose LLM")
    llm_choice = st.selectbox("Select LLM", ["Gemini", "Cohere", "GPT-3.5", "GPT-4"], on_change=reset_pipeline_generated)
    
    if llm_choice == "GPT-3.5":
        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo-1106")
        st.write(f"{llm_choice} selected")
    elif llm_choice == "GPT-4":
        llm = OpenAI(temperature=0.1, model="gpt-4-1106-preview")
        st.write(f"{llm_choice} selected")
    elif llm_choice == "Gemini":
        llm = Gemini(model="models/gemini-pro")
        st.write(f"{llm_choice} selected")
    elif llm_choice == "Cohere":
        llm = Cohere(model="command", api_key=os.environ['COHERE_API_TOKEN'])
        st.write(f"{llm_choice} selected")
    return llm, llm_choice

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
    selected_model = st.selectbox("Select Embedding Model", model_names,  on_change=reset_pipeline_generated)
    with st.spinner("Please wait") as status:
        embed_model = HuggingFaceEmbedding(model_name=selected_model)
        st.session_state['embed_model'] = embed_model
        st.markdown(F"Embedding Model: {embed_model.model_name}")
        st.markdown(F"Embed Batch Size: {embed_model.embed_batch_size}")
        st.markdown(F"Embed Batch Size: {embed_model.max_length}")


    return embed_model, selected_model

def select_node_parser():
    st.header("Choose Node Parser")
    parser_types = ["SentenceSplitter", "CodeSplitter", "SemanticSplitterNodeParser",
                    "TokenTextSplitter", "HTMLNodeParser", "JSONNodeParser", "MarkdownNodeParser"]
    parser_type = st.selectbox("Select Node Parser", parser_types, on_change=reset_pipeline_generated)
    
    parser_params = {}
    if parser_type == "HTMLNodeParser":
        tags = st.text_input("Enter tags separated by commas", "p, h1")
        tag_list = tags.split(',')
        parser = HTMLNodeParser(tags=tag_list)
        parser_params = {'tags': tag_list}
        
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
        parser_params = {'language': language, 'chunk_lines': chunk_lines, 'chunk_lines_overlap': chunk_lines_overlap, 'max_chars': max_chars}
        
    elif parser_type == "SentenceSplitter":
        chunk_size = st.number_input("Chunk Size", min_value=1, value=1024)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, value=20)
        parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        parser_params = {'chunk_size': chunk_size, 'chunk_overlap': chunk_overlap}
        
    elif parser_type == "SemanticSplitterNodeParser":
        if 'embed_model' not in st.session_state:
            st.warning("Please select an embedding model first.")
            return None, None
        
        embed_model = st.session_state['embed_model']
        buffer_size = st.number_input("Buffer Size", min_value=1, value=1)
        breakpoint_percentile_threshold = st.number_input("Breakpoint Percentile Threshold", min_value=0, max_value=100, value=95)
        parser = SemanticSplitterNodeParser(buffer_size=buffer_size, breakpoint_percentile_threshold=breakpoint_percentile_threshold, embed_model=embed_model)
        parser_params = {'buffer_size': buffer_size, 'breakpoint_percentile_threshold': breakpoint_percentile_threshold}
        
    elif parser_type == "TokenTextSplitter":
        chunk_size = st.number_input("Chunk Size", min_value=1, value=1024)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, value=20)
        parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        parser_params = {'chunk_size': chunk_size, 'chunk_overlap': chunk_overlap}

    # Save the parser type and parameters to the session state
    st.session_state['node_parser_type'] = parser_type
    st.session_state['node_parser_params'] = parser_params
    
    return parser, parser_type


def select_response_synthesis_method():
    st.header("Choose Response Synthesis Method")
    response_modes = [
        "refine",
        "tree_summarize",  
        "compact", 
        "simple_summarize", 
        "accumulate", 
        "compact_accumulate"
    ]
    selected_mode = st.selectbox("Select Response Mode", response_modes, on_change=reset_pipeline_generated)
    response_mode = selected_mode
    return response_mode, selected_mode

def select_vector_store():
    st.header("Choose Vector Store")
    vector_stores = ["Simple", "Pinecone", "Qdrant"]
    selected_store = st.selectbox("Select Vector Store", vector_stores, on_change=reset_pipeline_generated)

    vector_store = None

    if selected_store == "Pinecone":
        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        index = pc.Index("test")
        vector_store = PineconeVectorStore(pinecone_index=index)


    elif selected_store == "Qdrant":
        client = qdrant_client.QdrantClient(location=":memory:")
        vector_store = QdrantVectorStore(client=client, collection_name="sampledata")
    st.write(selected_store)
    return vector_store, selected_store

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
    response = f"Response for the query: {query}"
    st.markdown(response)

def generate_code_snippet(llm_choice, embed_model_choice, node_parser_choice, response_mode, vector_store_choice):
    node_parser_params = st.session_state.get('node_parser_params', {})
    print(node_parser_params)
    code_snippet = "from llama_index.llms import OpenAI, Gemini, Cohere\n"
    code_snippet += "from llama_index.embeddings import HuggingFaceEmbedding\n"
    code_snippet += "from llama_index import ServiceContext, VectorStoreIndex, StorageContext\n"
    code_snippet += "from llama_index.node_parser import SentenceSplitter, CodeSplitter, SemanticSplitterNodeParser, TokenTextSplitter\n"
    code_snippet += "from llama_index.node_parser.file import HTMLNodeParser, JSONNodeParser, MarkdownNodeParser\n"
    code_snippet += "from llama_index.vector_stores import MilvusVectorStore, QdrantVectorStore\n"
    code_snippet += "import qdrant_client\n\n"

    # LLM initialization
    if llm_choice == "GPT-3.5":
        code_snippet += "llm = OpenAI(temperature=0.1, model='gpt-3.5-turbo-1106')\n"
    elif llm_choice == "GPT-4":
        code_snippet += "llm = OpenAI(temperature=0.1, model='gpt-4-1106-preview')\n"
    elif llm_choice == "Gemini":
        code_snippet += "llm = Gemini(model='models/gemini-pro')\n"
    elif llm_choice == "Cohere":
        code_snippet += "llm = Cohere(model='command', api_key='<YOUR_API_KEY>')  # Replace <YOUR_API_KEY> with your actual API key\n"

    # Embedding model initialization
    code_snippet += f"embed_model = HuggingFaceEmbedding(model_name='{embed_model_choice}')\n\n"

    # Node parser initialization
    node_parsers = {
        "SentenceSplitter": f"SentenceSplitter(chunk_size={node_parser_params.get('chunk_size', 1024)}, chunk_overlap={node_parser_params.get('chunk_overlap', 20)})",
        "CodeSplitter": f"CodeSplitter(language={node_parser_params.get('language', 'python')}, chunk_lines={node_parser_params.get('chunk_lines', 40)}, chunk_lines_overlap={node_parser_params.get('chunk_lines_overlap', 15)}, max_chars={node_parser_params.get('max_chars', 1500)})",
        "SemanticSplitterNodeParser": f"SemanticSplitterNodeParser(buffer_size={node_parser_params.get('buffer_size', 1)}, breakpoint_percentile_threshold={node_parser_params.get('breakpoint_percentile_threshold', 95)}, embed_model=embed_model)",
        "TokenTextSplitter": f"TokenTextSplitter(chunk_size={node_parser_params.get('chunk_size', 1024)}, chunk_overlap={node_parser_params.get('chunk_overlap', 20)})",
        "HTMLNodeParser": f"HTMLNodeParser(tags={node_parser_params.get('tags', ['p', 'h1'])})",  
        "JSONNodeParser": "JSONNodeParser()",
        "MarkdownNodeParser": "MarkdownNodeParser()"
    }
    code_snippet += f"node_parser = {node_parsers[node_parser_choice]}\n\n"

    # Response mode
    code_snippet += f"response_mode = '{response_mode}'\n\n"

    # Vector store initialization
    if vector_store_choice == "Faiss":
        code_snippet += "d = 1536\n"
        code_snippet += "faiss_index = faiss.IndexFlatL2(d)\n"
        code_snippet += "vector_store = FaissVectorStore(faiss_index=faiss_index)\n"
    elif vector_store_choice == "Milvus":
        code_snippet += "vector_store = MilvusVectorStore(dim=1536, overwrite=True)\n"
    elif vector_store_choice == "Qdrant":
        code_snippet += "client = qdrant_client.QdrantClient(location=':memory:')\n"
        code_snippet += "vector_store = QdrantVectorStore(client=client, collection_name='sampledata')\n"
    elif vector_store_choice == "Simple":
        code_snippet += "vector_store = None  # Simple in-memory vector store selected\n"

    code_snippet += "\n# Finalizing the RAG pipeline setup\n"
    code_snippet += "if vector_store is not None:\n"
    code_snippet += "    storage_context = StorageContext.from_defaults(vector_store=vector_store)\n"
    code_snippet += "else:\n"
    code_snippet += "    storage_context = None\n\n"

    code_snippet += "service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)\n\n"

    code_snippet += "_file = 'path_to_your_file'  # Replace with the path to your file\n"
    code_snippet += "vector_index = VectorStoreIndex.from_documents(documents=_file, storage_context=storage_context, service_context=service_context, show_progress=True)\n"
    code_snippet += "if storage_context:\n"
    code_snippet += "    vector_index.storage_context.persist(persist_dir='persist_dir')\n\n"

    code_snippet += "query_engine = vector_index.as_query_engine(response_mode=response_mode, verbose=True)\n"

    return code_snippet

def main():
    st.title("RAGArch: RAG Pipeline Tester and Code Generator")
    st.markdown("""
    - **Configure and Test RAG Pipelines with Custom Parameters**
    - **Automatically Generate Plug-and-Play Implementation Code Based on Your Configuration**
    """)

    # Sidebar Intro
    st.sidebar.markdown('## App Created By')
    st.sidebar.markdown("""
    Harshad Suryawanshi: 
    [Linkedin](https://www.linkedin.com/in/harshadsuryawanshi/), [Medium](https://harshadsuryawanshi.medium.com/), [X](https://twitter.com/HarshadSurya1c)
    """)
    st.sidebar.markdown('## Other Projects')
    st.sidebar.markdown("""
    - [C3 Voice Assistant - Making LLM/RAG Apps Accessible to Everyone](https://www.linkedin.com/posts/harshadsuryawanshi_ai-llamaindex-gpt3-activity-7149796976442740736-1lXj?utm_source=share&utm_medium=member_desktop)
    - [NA2SQL - Extracting Insights from Databases using Natural Language](https://www.linkedin.com/posts/harshadsuryawanshi_ai-llamaindex-streamlit-activity-7141801596006440960-mCjT)
    - [Pokemon Go! Inspired AInimal GO! - Multimodal RAG App](https://www.linkedin.com/posts/harshadsuryawanshi_llamaindex-ai-deeplearning-activity-7134632983495327744-M7yy)
    - [Building My Own GPT4-V with PaLM and Kosmos](https://lnkd.in/dawgKZBP)
    - [AI Equity Research Analyst](https://ai-eqty-rsrch-anlyst.streamlit.app/)
    - [Recasting "The Office" Scene](https://blackmirroroffice.streamlit.app/)
    - [Story Generator](https://appstorycombined-agaf9j4ceit.streamlit.app/)
    """)
    
    st.sidebar.markdown('## Disclaimer')
    st.sidebar.markdown("""This application is for demonstration purposes only and may not cover all aspects of real-world data complexities. Please use it as a guide and not as a definitive source for decision-making.""")
    

    # Upload file
    file = upload_file()

    # Select RAG components
    llm, llm_choice = select_llm()
    embed_model, embed_model_choice = select_embedding_model()


    node_parser, node_parser_choice = select_node_parser()
    # Process nodes only if a file has been uploaded
    if file is not None:
        if node_parser:
            nodes = node_parser.get_nodes_from_documents(file)
            st.write("First node: ")
            st.code(f"{nodes[0].text}")

    response_mode, response_mode_choice = select_response_synthesis_method()
    vector_store, vector_store_choice = select_vector_store()

    # Generate RAG Pipeline Button
    if file is not None:
        if st.button("Generate RAG Pipeline"):
            with st.spinner():
                query_engine = generate_rag_pipeline(file, llm, embed_model, node_parser, response_mode, vector_store)
                st.session_state['query_engine'] = query_engine
                st.session_state['pipeline_generated'] = True
                st.success("RAG Pipeline Generated Successfully!")
    elif file is None:
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
  
    if file and st.button("Generate Code Snippet"):
        code_snippet = generate_code_snippet(llm_choice, embed_model_choice, node_parser_choice, response_mode_choice, vector_store_choice)
        st.code(code_snippet, language='python')

if __name__ == "__main__":
    main()
