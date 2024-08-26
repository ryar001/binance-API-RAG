from pydantic import BaseModel
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain import hub


class Prompts(BaseModel):
    '''Prompts for different use cases'''
    RAG_PROMPT:ChatPromptTemplate = ChatPromptTemplate.from_template("You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Include the sources used and its page number\nQuestion: {question} \nContext: {context} \nAnswer:")
    RAG_PROMPT_WITH_SOURCES:ChatPromptTemplate = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. 
        Include the sources used, their titles, and the page numbers.

        Question: {input}

        Context:
        {context}

        Answer:
        """)
    
    def pull_prompt(self, prompt_url:str="rlm/rag-prompt"):
        """
        Retrieves a prompt from the specified URL.

        Args:
            prompt_url (str): The URL of the prompt to retrieve. Defaults to "rlm/rag-prompt".

        Returns:
            The retrieved prompt.
        """
        return hub.pull(prompt_url)

    @staticmethod
    def create_prompt_from_text(template_text:str="",input_variables:List[str]=None):
        """
        Creates a PromptTemplate object from a given template text and input variables.

        Args:
            template_text (str): The template text with placeholders for input variables.
            input_variables (List[str]): A list of input variable names that correspond to the placeholders in the template text.

        Returns:
            PromptTemplate: A PromptTemplate object with the specified input variables and template text.
        """
        # Create a PromptTemplate object
        return PromptTemplate(
            input_variables=input_variables,  # Specify the placeholders in the text
            template=template_text  # The template text itself
        )