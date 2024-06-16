from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import load_chain
from langchain_community.llms import OpenAI
import getpass
import os
import dotenv


dotenv.load_dotenv(dotenv_path=".env")  # Specify the path to your .env_ai file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", default=getpass.getpass())


class RAGModel:
    def __init__(
        self, pdf_path, embedding_model="text-embedding-3-large", llm_model="gpt-4o"
    ):
        self.pdf_path = pdf_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.documents = None
        self.docs = None
        self.faiss_index = None
        self.llm = None
        self.rag_chain = None

    def load_pdf(self):
        loader = PyPDFLoader(self.pdf_path)
        self.documents = loader.load_and_split()

    def split_text(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Customize the chunk size based on your needs
            chunk_overlap=100,  # Overlap can be adjusted based on the structure
            separators=["\n\n", "\n", " "],
        )
        self.docs = text_splitter.split_documents(self.documents)

    def embed_text(self):
        breakpoint()
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.faiss_index = FAISS.from_documents(self.docs, embeddings)

    def initialize_llm(self):
        self.llm = OpenAI(
            model=self.llm_model,
            organization="org-jtbvxrXIr91IEWyUzMTHjBux",
            project="$PROJECT_ID",
        )

    def create_rag_chain(self):
        retriever = self.faiss_index.as_retriever()
        self.rag_chain = load_chain(
            {"type": "retrieval_qa", "llm": self.llm, "retriever": retriever}
        )

    def answer_question(self, question):
        if self.rag_chain:
            return self.rag_chain({"query": question})
        else:
            raise ValueError(
                "RAG chain is not initialized. Call create_rag_chain() first."
            )


if __name__ == "__main__":
    pdf_path = "./pdf_files/lamrim/lr-simplified-chinese02.pdf"
    rag_model = RAGModel(pdf_path)

    rag_model.load_pdf()
    rag_model.split_text()
    rag_model.embed_text()
    rag_model.initialize_llm()
    rag_model.create_rag_chain()

    while 1:
        question = input("Enter your question: ")
        answer = rag_model.answer_question(question)
        print(answer)
