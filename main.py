import getpass
import os
from typing import List, Dict, Any
from components.utils import *
from create_vectorstore import CreateVectorStore
from components.website.binance_api_docs import BinanceApiDocs
from components.rag_model import RAGModel
from components.prompts import Prompts



def main():
    # loading the VectorStore
    if LOAD_LOCAL:
        docs = BinanceApiDocs().get_all_endpoints()
        cc = CreateVectorStore(docs=[],mode="replace",filetype="nested_dict",vector_store_base_fp='components/vectore_indexes/')
        nest_dict_output_fp_list,_ = cc.recursive_get_nested_dict_docs_info(docs,make_dir=MAKE_DIR)
        local_vectorstore_dict = cc.load_all_vectore_stores(nest_dict_output_fp_list)
    merged_vs = cc.merge_all_vectore_stores(local_vectorstore_dict.values())

    # loading the RAG model
    rag = RAGModel(merged_vs,prompt=RAG_PROMPT_WITH_SOURCES,llm_api_key=API_KEY,include_metadata=INCLUDE_METADATA)

    # querying the RAG model
    while 1:
        question = input("Enter your query,enter 'exit' to quit: ")
        if question.lower() == "exit":
            break
        ans = rag.answer_question(question)
        print(ans) 
    breakpoint()

if __name__ == "__main__":
    MAKE_DIR = True
    LOAD_LOCAL = True
    RAG_PROMPT_WITH_SOURCES = Prompts().RAG_PROMPT_WITH_SOURCES
    API_KEY_VAR = "OPENAI_API_KEY"
    INCLUDE_METADATA = True

    API_KEY = os.getenv(API_KEY_VAR)
    if not API_KEY:
        API_KEY=getpass.getpass(prompt="llm Api key: ")

    # start the rag model
    main()