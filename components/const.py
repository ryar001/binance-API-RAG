from langchain_community import embeddings
from langchain_community import  vectorstores
from utils import MetaClass
from typing import Any,List,Dict


class Embeddings(metaclass=MetaClass):
    '''Embeddings class'''
    BASE = embeddings 

class VectorStores(metaclass=MetaClass):
    '''VectorStores class'''
    BASE = vectorstores


if __name__ == "__main__":
    Embeddings
    breakpoint()
    print(Embeddings['open_ai_embeddings'])
    print(Embeddings.hugging_face_embeddings)
    print(Embeddings.__get_dict__())
    breakpoint( )
# Compare this snippet from components/chain.py:
# Compare this snippet from components/const.p