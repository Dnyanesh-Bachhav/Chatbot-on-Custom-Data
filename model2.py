import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    embeddings_folder = "./embeddings"

    # Load all Chroma vector stores
    # for file_name in os.listdir(embeddings_folder):
    #     if file_name.endswith('.chroma'):
    #         vectorstore_path = os.path.join(embeddings_folder, file_name)
    vector_stores = Chroma(persist_directory=embeddings_folder, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
    # vector_stores.get()
    # Combine vector stores if needed (optional)
    # For simplicity, use only the first vector store
    retriever = vector_stores.as_retriever()

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model="llama3:instruct"),
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg = cl.Message(content="System is ready. You can now ask questions!")
    await msg.send()

    # Store the chain in user session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")

    # Callbacks happen asynchronously/parallel
    cb = cl.AsyncLangchainCallbackHandler()

    # Call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])

    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []  # Initialize list to store text elements

    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    # Return results
    await cl.Message(content=answer, elements=text_elements).send()