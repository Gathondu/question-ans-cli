import os
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, WikipediaLoader


def init_pinecone():
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))


def delete_pinecone_indexes(index_name="all"):
    indexes = pinecone.list_indexes()

    if index_name == "all":
        print("Deleting all indexes ...", end="\n")
        if indexes:
            for i in indexes:
                print(f"deleting index {i}", end=" ")
                pinecone.delete_index(i)
                print("Done", end="\n")
            print("All indexes deleted")
        else:
            print("No indexes found.")
    else:
        print(f"Deleting index {index_name}", end=" ")
        pinecone.delete_index(index_name)
        print("Done")


def load_document(file: str):
    print(f"Loading {file} ...", end=" ")
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    print("Done")
    return pages


def load_from_wikipedia(query: str, lang="en", load_max_docs=2):
    print(f"Fetching {query} from wikipedia..")
    return WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs).load()


def create_chunks(data, chunk_size=256) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_documents(data)


def print_embedding_cost(chunks: list):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(enc.encode(page.page_content)) for page in chunks])
    print(f"Total Tokens: {total_tokens}", end="\n")
    print(f"Embedding Cost in USD: {total_tokens/1000 * 0.0004:.6f}")


def insert_or_fetch_embeddings(index_name: str, chunks: list):
    embeddings = OpenAIEmbeddings()
    if index_name in pinecone.list_indexes():
        print(f"Index {index_name} already exists...loading embbeddings ...", end="")
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
    else:
        delete_pinecone_indexes()
        print(f"Creating index {index_name} and embeddings ...", end="")
        pinecone.create_index(index_name, dimension=1536, metric="cosine")
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    print("Ok")
    return vector_store


def ask_and_get_answer(vector_store, query: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return chain.run(query)


if __name__ == "__main__":
    import time

    load_dotenv(find_dotenv(), override=True)
    index_name = "constitution"
    filename = "files/constitution.pdf"
    data = load_document(filename)
    # Create chunks and embeddings
    chunks = create_chunks(data, chunk_size=300)  # setting this high to reduce the number of chunks for the document
    print_embedding_cost(chunks)
    # Add the vectors to pinecone. We are only allowed one index in free tier
    init_pinecone()
    vector_store = insert_or_fetch_embeddings(index_name, chunks)

    i = 1  # to track the questions
    print("Write Quit or Exit to quit.")
    while True:
        q = input(f"Question #{i}: ")
        i += 1
        if q.lower() in ["quit", "exit"]:
            print("Quiting ... bye bye!")
            time.sleep(2)
            break
        answer = ask_and_get_answer(vector_store, q)
        print(f"\nAnswer: {answer}")
        print(f'\n {"_"*50} \n')
