from langchain_community import embeddings
from langchain_community import vectorstores
from langchain_community import document_loaders
from langchain_community import retrievers
from langchain_openai import OpenAIEmbeddings
from typing import Dict, Any, Tuple


class MetaClass(type):
    EMBEDDINGS_SPECIAL_MAPPINGS:Dict[str,str] = {
        'openai_embeddings':OpenAIEmbeddings
    }
    def __get_dict__(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if not key.startswith('__')}

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __new__(cls, name:str, bases:Tuple, dct:Dict[str,Any]):
        # Create the new class
        new_class = super().__new__(cls, name, bases, dct)

        # Enforce that the class has a BASE attribute
        if not hasattr(new_class, 'BASE'):
            raise AttributeError(f"Class {name} must have a 'BASE' attribute from langchain_community")

        # Access BASE from the class
        base_instance = new_class.BASE

        # Set the attributes of the new class
        for key,val in base_instance._module_lookup.items():        
            obj_name = f"{val.split('.')[-1]}_{val.split('.')[-2]}"
            _obj_attr = cls.EMBEDDINGS_SPECIAL_MAPPINGS.get(obj_name) if cls.EMBEDDINGS_SPECIAL_MAPPINGS.get(obj_name) else getattr(base_instance,key)
            setattr(new_class,val.split('.')[-1],_obj_attr) 
        return new_class
    
    def _set_openai_embeddings(self):
        self.open_ai_embeddings = OpenAIEmbeddings


class Embeddings(metaclass=MetaClass):
    '''Embeddings class'''
    BASE = embeddings

class VectorStores(metaclass=MetaClass):
    '''VectorStores class'''
    BASE = vectorstores

class DocumentLoaders(metaclass=MetaClass):
    '''DocumentLoaders class'''
    BASE = document_loaders

class Retriever(metaclass=MetaClass):
    '''Retriever class'''
    BASE = retrievers

if __name__ == "__main__":
    Embeddings
    breakpoint()
    print(Embeddings['open_ai_embeddings'])
    print(Embeddings.hugging_face_embeddings)
    print(Embeddings.__get_dict__())
    breakpoint( )
# Compare this snippet from components/chain.py:
# Compare this snippet from components/const.p