import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

async def prepare_embeddings():
    data_folder = "./data"
    embeddings_folder = "./embeddings"

    # Ensure the embeddings folder exists
    os.makedirs(embeddings_folder, exist_ok=True)

    # Process each PDF file in the data folder
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(data_folder, file_name)

            # Read the PDF file
            pdf = PyPDF2.PdfReader(file_path)
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text()

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
            texts = text_splitter.split_text(pdf_text)

            # Create metadata for each chunk
            metadatas = [{"source": f"{file_name}-{i}"} for i in range(len(texts))]

            # Generate embeddings
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = Chroma.from_texts(texts=texts, persist_directory=embeddings_folder, embedding=embeddings, metadatas=metadatas)

            # Save the vectorstore
            vectorstore.persist()
            # vectorstore.persist(os.path.join(embeddings_folder, f"{file_name}.chroma"))

    print("Embeddings prepared and saved.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(prepare_embeddings())
