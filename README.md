    # Use the parser to get nodes from documents
    nodes = parser.get_nodes_from_documents(file)
    if(nodes):
        print(nodes[0].text)
        #print(nodes[1].text)




nodes = parser.get_nodes_from_documents(file)

        # Initialize the language model
    llm = OpenAI(model_name="gpt-3.5-turbo")

    # Create the service context
    service_context = ServiceContext.from_defaults(llm=llm, node_parser=parser)

    # Create the vector index
    vector_index = VectorStoreIndex.from_documents(documents=file, service_context=service_context, show_progress=True)
    vector_index.storage_context.persist(persist_dir="persist_dir")

    query_engine = vector_index.as_query_engine()
    response = query_engine.query(
    "provide 1 paragraph summary."
    )
    print(response)
