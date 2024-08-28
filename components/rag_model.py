from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.base import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import PromptTemplate
import dotenv
import os
import getpass
from pathlib import Path
from typing import List, Dict
from typing_extensions import Annotated, TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field

BASE_PATH = Path(__file__).parent.parent

# dotenv.load_dotenv(dotenv_path=Path(BASE_PATH,".env")  ) # Specify the path to your .env_ai file
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", default=getpass.getpass(prompt="llm Api key: "))

class RAGModel:
    def __init__(
        self, vectore_store_obj=None, 
        prompt:PromptTemplate=None,
        llm_model:str="gpt-4o-mini-2024-07-18",
        llm_api_key:str=None,
        temperature:float=0,
        **kwargs):
        """
        Initializes a RAGModel instance with the given parameters.

        Args:
            vectore_store_obj: The vector store object to use.
            prompt: The prompt template to use. Defaults to None.
            embedding_model: The embedding model to use. Defaults to "text-embedding-3-large".
            llm_model: The LLM model to use. Defaults to "gpt-4o-mini-2024-07-18".
            temperature: The temperature to use. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.vector_stores_obj = vectore_store_obj
        self.prompt = prompt if prompt else hub.pull("rlm/rag-prompt")
        self.llm_model = llm_model
        self.temperature = temperature
        self.llm_api_key = llm_api_key
        self.documents = None
        self.docs = None
        self.llm = None
        self.vector_store = None
        self.rag_chain:RunnableSequence = None
        
        self.debug:bool = kwargs.get("debug", False)
        self.input_variables = kwargs.get("input_variables",self.prompt.input_variables)
        self.include_metadata:bool = kwargs.get("include_metadata",False)

        # get the input variables
        self.input_variables = self.get_prompt_input_variables(self.input_variables)
        
        # init the rag chain
        self.get_rag_chain()

    def initialize_llm(self,api_key=None,base_url=None,organization=None):
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            # other params...
        )
    @staticmethod
    def format_docs(docs,include_metadata:bool=False):
        _output = ""
        for doc in docs:
            _output += "\n\n"
            if include_metadata:
                for key,value in doc.metadata.items():
                    _output += f"{key}: {value}\n"
            _output += doc.page_content
        return _output
        # return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def get_prompt_input_variables(input_variables:List[str],exclude:str="context")->Dict[str,RunnablePassthrough]:
        return {i:RunnablePassthrough() for i in input_variables if i != exclude}
    
    @staticmethod
    def get_metadata(docs,key:str):
        return [doc.metadata.get(key) for doc in docs]

    def create_rag_chain(self):
        # Retriever is a simple retriever that uses the vector store object from langchain
        retriever = self.vector_stores_obj.as_retriever()
        self.rag_chain = (
            retriever
            | (lambda docs: {
                "context": self.format_docs(docs,include_metadata=self.include_metadata),
                "input": RunnablePassthrough(),
            })
            | self.prompt
            | self.llm.with_structured_output(AnswerWithSources)
        )

    def get_rag_chain(self):
        '''run all the methods to create the RAG chain'''
        self.initialize_llm()
        self.create_rag_chain()
    
    def answer_question(self, question:str):
        if self.rag_chain:
            return self.rag_chain.invoke(  question)
        else:
            raise ValueError(
                "RAG chain is not initialized. Call create_rag_chain() first."
            )
    
class AnswerWithSources(BaseModel):
    answer: str = Field(description="The answer from llm")
    url_link: str = Field(description="HTTP Url use for the context sent to llm")

if __name__ == "__main__":
    from website.binance_api_docs import BinanceApiDocs
    breakpoint()
    # sele_web_rag = RAGModel(debug=True,loader_name="SeleniumURLLoader" ,doc_path=["https://doc.xt.com/"])
    sele_web_rag = RAGModel(debug=True,loader_name="url_selenium",
                            vector_store_fp="/Users/jokerssd/Documents/RAG-freshstart/components/vectore_indexes/index.faiss",
                            )
    # sele_web_rag = RAGModel(debug=True,loader_name="url_selenium"
    #                         ,doc_path=["https://www.binance.com/en/futures/trading-rules/perpetual/portfolio-margin/collateral-ratio"],
    #                          vector_store_fp="/Users/jokerssd/Documents/RAG-freshstart/components/vectore_indexes/index.faiss")
    

    sele_web_rag_chain = sele_web_rag.get_rag_chain()
    while 1:
        question = input("Enter your query: ")
        ans = sele_web_rag_chain.answer_question(question)
        print(ans)    


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