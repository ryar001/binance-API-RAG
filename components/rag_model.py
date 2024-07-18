import utils
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain import text_splitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import load_chain
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import utils as const
import os
import dotenv
from pathlib import Path
from typing import List, Dict
from vectorstore_utils import VectorStoreUtils

BASE_PATH = Path(__file__).parent.parent

dotenv.load_dotenv(dotenv_path=Path(BASE_PATH,".env")  ) # Specify the path to your .env_ai file
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", default=getpass.getpass())



class RAGModel:
    RAG_PROMPT = ChatPromptTemplate.from_template("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Include the sources used and its page number\nQuestion: {question} \nContext: {context} \nAnswer:")

    def __init__(
        self, doc_path="", embedding_model="text-embedding-3-large", llm_model="gpt-4o",
        loader_name="PyPDFLoader",**kwargs):

        self.loader_name = loader_name
        self.doc_path = doc_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.documents = None
        self.docs = None
        self.faiss_index = None
        self.llm = None
        self.vector_store = None
        self.rag_chain:RunnableSequence = None
        self.prompt:ChatPromptTemplate = kwargs.get("prompt", self.RAG_PROMPT ) if kwargs.get("prompt", self.RAG_PROMPT ) else hub.pull("rlm/rag-prompt")
        self.debug:bool = kwargs.get("debug", False)
        self.temperature:float = kwargs.get("temperature", 0)
        self.splitter = kwargs.get("splitter", RecursiveCharacterTextSplitter)
        self.documents_loader: const.DocumentLoaders = const.DocumentLoaders
        self.vector_store_utils = VectorStoreUtils(**kwargs)
        # self.retriever = con

    def load_documents(self,urls:List[str]=[""], loader_name:str="",**kwargs):
        '''load documents from the specified loader'''
        if not loader_name:
            loader_name = self.loader_name
        loader = self.documents_loader.__dict__.get(loader_name)
        loader = loader(self.doc_path, **kwargs)
        self.documents = loader.load_and_split()
        return self.documents

    def pdf_loader(self,urls:List[str]=None,**kwargs):
        '''PyPDFLoader is a loader that uses PyPDF2 to load PDF files. It is useful for extracting text from PDF files.'''
        urls = urls if urls else self.doc_path
        loader = PyPDFLoader(urls)
        self.documents = loader.load_and_split()
        if self.debug:
            self.documents = self.documents[:2]

    def bshtml_loader(self,urls:List[str]=["https://binance-docs.github.io/apidocs/pm/en/#change-log"],**kwargs):
        '''BSHTMLLoader is a loader that uses BeautifulSoup to parse the HTML content of a webpage. It is useful for extracting text from webpages that are not easily accessible via the browser.'''
        loader = BSHTMLLoader(urls)
        self.documents = loader.load_and_split()
        return self.documents
        # print(docs[0].page_content[:61])
    
    def selenium_loader(self,urls:List[str]=["https://binance-docs.github.io/apidocs/pm/en/#change-log"],**kwargs):
        '''SeleniumURLLoader is a loader that uses Selenium to load webpages. It is useful for extracting text from webpages that require JavaScript to render.'''
        loader = SeleniumURLLoader(urls=urls)
        self.documents = loader.load_and_split()
        return self.documents

    def split_text(self):
        text_splitter = self.splitter(
            chunk_size=1000,  # Customize the chunk size based on your needs
            chunk_overlap=100,  # Overlap can be adjusted based on the structure
            separators=["\n\n", "\n", " "],
        )
        self.docs = text_splitter.split_documents(self.documents)

    def embed_text(self):
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.faiss_index = FAISS.from_documents(self.docs, embeddings)

    def initialize_llm(self):
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # base_url="...",
            # organization="...",
            # other params...
        )

    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def include_source(self):
        self.prompt

    def create_rag_chain(self):
        retriever = self.faiss_index.as_retriever()

        # RunnablePassthrough is a simple runnable that passes the input to the output , eg, the question to the output
        # StrOutputParser is a simple output parser that just output the str of the llm output
        self.rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return self.rag_chain
        # for chunk in self.rag_chain.stream("What is Task Decomposition?"):
        #     print(chunk, end="", flush=True)

    def answer_question(self, question):
        if self.rag_chain:
            return self.rag_chain.invoke( question)
        else:
            raise ValueError(
                "RAG chain is not initialized. Call create_rag_chain() first."
            )
    
    def get_vectorstore(self,docs):
        self.vector_store = self.vector_store_utils.get_vectorstore(data_type="documents",text_chunks=docs,mode="merge")
        
    def create_chain(self):
        '''run all the methods to create the RAG chain'''
        self.load_documents(self.loader_name)
        self.split_text()
        self.embed_text()
        self.get_vectorstore(self.docs)
        self.initialize_llm()
        self.create_rag_chain()
        return self.rag_chain
    
    # def add_to
    

if __name__ == "__main__":
    # sele_web_rag = RAGModel(debug=True,loader_name="SeleniumURLLoader" ,doc_path=["https://doc.xt.com/"])
    sele_web_rag = RAGModel(debug=True,loader_name="url_selenium"
                            ,doc_path=["https://doc.xt.com/#documentationrestApi"],
                             vector_store_fp="/Users/jokerssd/Documents/RAG-freshstart/components/vectore_indexes/index.faiss",
                             )
    # sele_web_rag = RAGModel(debug=True,loader_name="url_selenium"
    #                         ,doc_path=["https://www.binance.com/en/futures/trading-rules/perpetual/portfolio-margin/collateral-ratio"],
    #                          vector_store_fp="/Users/jokerssd/Documents/RAG-freshstart/components/vectore_indexes/index.faiss")
    
    sele_web_rag_chain = sele_web_rag.create_chain()
    breakpoint()


    # base_path = Path(__file__).parent.parent
    # pdf_path = Path(base_path,"pdf_files/lamrim/lr-simplified-chinese02.pdf")
    # rag_recur = RAGModel(pdf_path,debug=True )
    # rag_seman = RAGModel(pdf_path,debug=True, splitter=SemanticChunker)
    # rag_spacy = RAGModel(pdf_path,debug=True, splitter=text_splitter.SpacyTextSplitter)

    # rc_recur = rag_recur.create_chain()
    # rc_seman = rag_seman.create_chain()
    # rc_spacy = rag_spacy.create_chain()

    # while 1:
    #     question = input("Enter your question: ")
    #     ans_recur = rc_recur.answer_question(question)
    #     ans_semen = rc_seman.answer_question(question)
    #     ans_spacy = rc_spacy.answer_question(question)
    #     print(f"""Recursive: {ans_recur}\n
    #           Semantic: {ans_semen}\n
    #           Spacy: {ans_spacy}""")