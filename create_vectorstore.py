import random
import string
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from components.utils import *

class CreateVectorStore:
    def __init__(self,docs,mode="merge",docs_type="url",filetype="nested_dict",vector_store_base_fp='components/vectore_indexes/',**kwargs):
        self.docs = docs
        self.loaded_docs_dict = {}
        self.docs_type = docs_type
        self.mode = mode
        self.vector_store_base_fp = vector_store_base_fp
        self.vector_store = None
        self.vectorstore_dict = {}
        self.vector_store_fp = None
        self.vector_store_indexes = {}
        self.faiss_index = None
        self.nest_dict_output_fp_list = []
        self.keys_path = ""
        self.filetype = filetype
        self.splitter = kwargs.get("splitter", RecursiveCharacterTextSplitter)
        self.loader_name = kwargs.get("loader_name", "url_selenium")
        self.documents_loader: const.DocumentLoaders = const.DocumentLoaders
        self.vector_store_utils = VectorStoreUtils()
        self.multi_thread_utils = MultiThreadUtils()

    def recursive_get_nested_dict_docs_info(self,nested_dict:dict,**kwargs):
        '''iterate over the nested dictionary and create a list of documents, and create , save to vectorstore and return the vectorstore_fp'''
        def _recursive_get_nested_dict_docs_info(nested_dict:dict,keys_path,**kwargs):
            make_dir = kwargs.get("make_dir", False)
            for key, value in nested_dict.items():
                if isinstance(value, dict):
                    '''depth search into each dict'''
                    _recursive_get_nested_dict_docs_info(value,Path(keys_path, key),**kwargs)
                else:
                    '''leaf node, append the path and create the dir'''
                    fp = str(Path(self.vector_store_base_fp,keys_path,key))
                    self.nest_dict_output_fp_list.append(fp.lower())
                    if make_dir:
                        Path(fp).mkdir(parents=True, exist_ok=True)
                    self.docs.append(value)
            return self.nest_dict_output_fp_list,self.docs
        keys_path = ""
        return _recursive_get_nested_dict_docs_info(nested_dict,keys_path,**kwargs)
    
    def get_today_date(self):
        import datetime
        return datetime.date.today()

    def get_random_id(self):
        '''get a random str id'''
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    
    def _load_n_store(self,fp,doc_url,loader_name):
        '''create vectorstore and save to vectorstore_fp'''
        vector_store_utils = VectorStoreUtils(vector_store_fp=fp,loader_name=loader_name)
        self.loaded_docs_dict[fp] = vector_store_utils.load_documents(doc_path=[doc_url],just_load=True)
        return vector_store_utils.get_vectorstore(data_type="documents",text_chunks=self.loaded_docs_dict[fp],mode=self.mode)

    def multithread_load(self,nest_dict_output_fp_list,docs_url_list,loader_name,**kwargs)->Dict[str,Any]:
        """
        Loads documents from a list of file paths and URLs in parallel using multiple threads.

        Args:
            nest_dict_output_fp_list (List[str]): A list of file paths where the documents will be saved.
            docs_url_list (List[str]): A list of URLs from which the documents will be loaded.
            loader_name (str): The name of the loader to be used for loading the documents.
            **kwargs: Additional keyword arguments that will be passed to the loader.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the thread executions.

        Raises:
            None

        Side Effects:
            - Creates vector stores at the specified file paths.
            - Prints the progress of document loading and saving.
        """

        # create vectorstore
        for fp,doc_url in zip(nest_dict_output_fp_list,docs_url_list):
            print(f"Pulling docs from {doc_url} and saving to {fp}")
            # doc = vector_store_utils.load_documents(doc_path=[doc_url],just_load=True)                
            self.vector_store_indexes[fp] = self.multi_thread_utils.create_thread(name=fp, target=self._load_n_store,
                                                                                  args=[],kwargs={"fp":fp,"doc_url":doc_url,
                                                                                                  "loader_name":loader_name,
                                                                                                  })
            print(f"Done pulling docs from {doc_url} and saving to {fp}\n")

        # start all threads
        self.multi_thread_utils.start_all()

        # wait for all threads to finish
        self.multi_thread_utils.join_all()

        # return docs
        return  self.multi_thread_utils.get_all_thread_results()

    def create_nested_dict_vectorstore(self,nested_dict, loader_name:str="",**kwargs):
        # TODO: add in the multi thread option
        make_dir = kwargs.get("make_dir", True)
        do_multithread = kwargs.get("do_multithread", False)
        debug = kwargs.get("debug", False)
        nest_dict_output_fp_list = kwargs.get("nest_dict_output_fp_list", [])
        docs_url_list = kwargs.get("docs_url_list", [])

        if not loader_name:
            loader_name = self.loader_name
        if not nest_dict_output_fp_list or not docs_url_list:
            # iterate over the nested dictionary and create a list of documents, and create , save to vectorstore and return the vectorstore_fp
            nest_dict_output_fp_list,docs_url_list = self.recursive_get_nested_dict_docs_info(nested_dict,make_dir=make_dir)

        if debug:
            nest_dict_output_fp_list = nest_dict_output_fp_list[0:1]
            docs_url_list = docs_url_list[0:1]
        
        # run with multithreading
        if do_multithread:
            self.vectorstore_dict = self.multithread_load(nest_dict_output_fp_list,docs_url_list,loader_name=loader_name)
            return self.vectorstore_dict

        # create vectorstore
        for fp,doc_url in zip(nest_dict_output_fp_list,docs_url_list):
            vector_store_utils = VectorStoreUtils(vector_store_fp=fp,loader_name=loader_name)
            doc = vector_store_utils.load_documents(doc_path=[doc_url],just_load=True)
            self.docs.append(doc)
            # self.vector_store_utils.get_vectorstore(data_type="text",text_chunks=[doc_url],mode="merge")

        self.vector_store = self.vector_store_utils.get_vectorstore(data_type="documents",text_chunks=self.docs,mode="merge")
        self.vector_store_fp = self.vector_store_utils.save_vectorstore(self.vector_store)
        return self.vector_store_fp
    
    def merge_vectorstore(self,v_dict:dict):
        '''merge all the vectorstores in v_dict and return the merged vectorstore'''

    def load_all_vectore_stores(self,fp_list:List[str],)->dict:
        '''load all vectorstores in fp_list'''
        for fp in fp_list:
            self.vectorstore_dict[fp] = self.vector_store_utils.load_vectorstore(fp,allow_dangerous_deserialization=True)
        return self.vectorstore_dict
    
    def merge_all_vectore_stores(self,v_list:list):
        '''merge all the vectorstores in v_dict and return the merged vectorstore'''
        breakpoint()
        for vec in v_list:
            self.vector_store = self.vector_store_utils.merge_vectorstore(self.vector_store,vec)
           
if __name__ == "__main__":
    from components.website.binance_api_docs import BinanceApiDocs

    MAKE_DIR = True

    docs = BinanceApiDocs().get_all_endpoints()
    lefty = {"QUERY_CURRENT_CM_OPEN_ORDERS":'https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-Current-CM-Open-Orders',
             "QUERY_CURRENT_UM_OPEN":'https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-Current-UM-Open',
             "QUERY_UM_CONDITIONAL_ORDER":'https://developers.binance.com/docs/derivatives/portfolio-margin/trade/Query-UM-Conditional-Order'}
    breakpoint()
    cc = CreateVectorStore(docs=[],mode="merge",filetype="nested_dict",vector_store_base_fp='components/vectore_indexes/')
    nest_dict_output_fp_list,docs_url_list = cc.recursive_get_nested_dict_docs_info(lefty,make_dir=MAKE_DIR)
    # local_vectorstore_dict = cc.load_all_vectore_stores(nest_dict_output_fp_list)
    # merged_vs = cc.merge_all_vectore_stores(local_vectorstore_dict.values())
    breakpoint()
    res = cc.create_nested_dict_vectorstore(lefty,debug=False,do_multithread=True,
                                            nest_dict_output_fp_list=nest_dict_output_fp_list,docs_url_list=docs_url_list)
    breakpoint()