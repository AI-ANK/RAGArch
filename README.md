
# RAGArch: Configure/Test Llamaindex RAG Pipelines and One-click Autogenerate Plug-n-play Code

RAGArch is a Streamlit-based application that empowers users to experiment with various components and parameters of Retrieval-Augmented Generation (RAG) pipelines. Utilizing the power of Llamaindex, RAGArch facilitates the testing of different configurations to see how they perform. Once satisfied, users can generate the Python code for their custom RAG pipeline configurations, enabling easy integration into their applications.

## Features

- **Interactive UI:** Test different RAG pipeline components through an intuitive web interface.
- **Custom Configurations:** Choose from various Large Language Models, Embedding Models, Node Parsers, Response Synthesis Methods, and Vector Stores.
- **Live Testing:** Upload your data and immediately see how your RAG pipeline performs.
- **Code Generation:** Generate and export the Python code for your configured pipeline with a single click.

## How to Use

1. **Set Up Your Environment:**
   Ensure you have Python and Streamlit installed. Clone the repository and install the required dependencies.

2. **Launch the Application:**
   Run the app with Streamlit:

   ```bash
   streamlit run app.py
   ```

3. **Configure Your Pipeline:**
   - Upload your data file.
   - Select your preferred Language Model, Embedding Model, and other pipeline components.
   - Click "Generate RAG Pipeline" to test the pipeline with your data.

4. **Generate Code:**
   After testing, click "Generate Code Snippet" to receive the Python code for your custom configuration.

## Installation

```bash
git clone https://github.com/your-username/RAGArch.git
cd RAGArch
pip install -r requirements.txt
```

## Tools and Technologies

- UI: Streamlit
- LLM Orchestration: Llamaindex
- LLMs: OpenAI GPT 3.5 and 4, Cohere API, Gemini Pro
- Embedding Models:
   - "BAAI/bge-small-en-v1.5"
   - "WhereIsAI/UAE-Large-V1"
   - "BAAI/bge-large-en-v1.5"
   - "khoa-klaytn/bge-small-en-v1.5-angle"
   - "BAAI/bge-base-en-v1.5"
   - "llmrails/ember-v1"
   - "jamesgpt1/sf_model_e5"
   - "thenlper/gte-large"
   - "infgrad/stella-base-en-v2"
   - "thenlper/gte-base" 
- Vector Stores: Simple, Pinecone, Qdrant

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Support

For support, please open an issue in the GitHub issue tracker.

## Live Demo
[Try the demo here!](https://huggingface.co/spaces/AI-ANK/RAGArch)


## Authors
### Developed by [Harshad Suryawanshi](https://www.linkedin.com/in/harshadsuryawanshi/)
If you find this project useful, consider giving it a ⭐ on GitHub!
