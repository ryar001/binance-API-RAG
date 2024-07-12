from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStore as VectorStoreClass
from langchain_core.embeddings import Embeddings as EmbeddingsClass
from langchain_community.callbacks.manager import get_openai_callback
from const import Embeddings, VectorStores
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env" )

class VectoreStore:
    """
    A class representing a vector store.

    Attributes:
        vector_store_fp (str): The file path of the vector store.
        embeddings (EmbeddingsClass): The embeddings class used for vectorization.
        vectorstore (VectorStoreClass): The vector store class used for storing vectors.

    Methods:
        create_vectorstore: Creates a vector store from text chunks.
        load_vectorstore: Loads a vector store from a file.
        get_vectorstore: Retrieves the vector store.
        merge_vectorstore: Merges two vector stores.
        search_docs: Searches for documents in the vector store.
        get_retriever: Retrieves a retriever object from the vector store.
    """

    def __init__(self, vector_store_fp: str = "vectorstore.pkl", vectorstore: str = "faiss", embeddings: str = "openai"):
        self.vector_store_fp = vector_store_fp
        self.embeddings: EmbeddingsClass = Embeddings[embeddings]
        self.vectorstore: VectorStoreClass = VectorStores[vectorstore]

    def create_vectorstore(self, text_chunks: list, data_type: str = "text", **kwargs):
        """
        Creates a vector store from text chunks.

        Args:
            text_chunks (list): A list of text chunks.
            data_type (str): The type of data. Default is "text".
            **kwargs: Additional keyword arguments.

        Returns:
            tmp_vectorstore: The created vector store.
        """
        embeddings = kwargs.get("embeddings", self.embeddings)
        tmp_vectorstore = None
        with get_openai_callback() as cb:
            if data_type == "text":
                tmp_vectorstore = self.vectorstore.from_texts(
                    texts=text_chunks, embedding=embeddings)
            if data_type == "documents":
                tmp_vectorstore = self.vectorstore.from_documents(
                    documents=text_chunks, embedding=self.embeddings
                )
            print(cb)

        return tmp_vectorstore

    def load_vectorstore(self, vs_path: str, embeddings=None, **kwargs):
        """
        Loads a vector store from a file.

        Args:
            vs_path (str): The file path of the vector store.
            embeddings: The embeddings class used for vectorization.
            **kwargs: Additional keyword arguments.

        Returns:
            vectorstore: The loaded vector store.
        """
        embeddings = kwargs.get("embeddings", self.embeddings)

        if not Path(self.vector_store_fp).exists():
            return None

        return self.vectorstore.load_local(vs_path, embeddings=embeddings)

    def get_vectorstore(self, data_type: str = "text", text_chunks: list = None, mode: str = "merge", **kwargs):
        """
        Retrieves the vector store.

        Args:
            data_type (str): The type of data. Default is "text".
            text_chunks (list): A list of text chunks.
            mode (str): The mode of operation. Default is "merge".
            **kwargs: Additional keyword arguments.

        Returns:
            self.vectorstore: The vector store.
        """
        embeddings = kwargs.get("embeddings", self.embeddings)
        vector_store_fp = kwargs.get("vector_store_fp", self.vector_store_fp)

        if Path(vector_store_fp).exists():
            self.vectorstore = self.load_vectorstore(
                vector_store_fp, embeddings=embeddings)
            print("Loaded vectorstore")

        # check if additional text_chunks are provided
        # if not will return the existing vectorstore
        if not text_chunks:
            return self.vectorstore

        # create vectorstore and print the cost
        tmp_vectorstore = self.create_vectorstore(
            text_chunks=text_chunks, data_type=data_type, embeddings=embeddings)

        # merge Vector store
        if self.vectorstore and tmp_vectorstore and mode == "merge":
            self.merge_vectorstore(self.vectorstore, tmp_vectorstore)

        # save vectorstore to local
        self.vectorstore.save_local(vector_store_fp)
        print("vectorstore created and saved")
        return self.vectorstore

    def merge_vectorstore(self, vectorstore: VectorStoreClass, vectorstore2: VectorStoreClass)-> VectorStoreClass:
        """
        Merges two vector stores.

        Args:
            vectorstore (VectorStoreClass): The first vector store.
            vectorstore2 (VectorStoreClass): The second vector store.

        Returns:
            vectorstore (VectorStoreClass): The merged vector store.
        """
        vectorstore.merge_from(vectorstore2)
        return vectorstore

    def search_docs(self, query: str, method: str = "similarity_search", top_x: int = 5, **kwargs):
        """
        Searches for documents in the vector store.

        Args:
            query (str): The query string.
            method (str): The search method. Default is "similarity_search".
            top_x (int): The number of top results to return. Default is 5.
            **kwargs: Additional keyword arguments.

        Returns:
            results: The search results.
        """
        vector_store: VectorStoreClass = kwargs.get("vectorstore", self.vectorstore)
        if method not in dir(vector_store):
            return None
        return getattr(vector_store, method)(query)[:top_x]

    def get_retriever(self, search_kwargs: Dict[str: str] = None, **kwargs):
        """
        Retrieves a retriever object from the vector store.

        Args:
            search_kwargs (Dict[str: str]): The search keyword arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            retriever: The retriever object.
        """
        vectorstore: VectorStoreClass = kwargs.get('vectorstore', self.vectorstore)
        return vectorstore.as_retriever(search_kwargs=search_kwargs, **kwargs)

if __name__ =="__main__":
    vs = VectoreStore()
    vs.get_vectorstore(text_chunks=["hello world"],data_type="text")
    vs.search_docs(vs.vectorstore,"hello world")
    breakpoint()
    print("done")