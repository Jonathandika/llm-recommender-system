from dotenv import dotenv_values

import pinecone

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch
from langchain.agents import Tool
from langchain.agents import initialize_agent

from helper.PineconeModified import PineconeModified
from helper.PineconeSelfQueryRetriever import PineconeSelfQueryRetriever

import os

class RAG:
    MODEL_NAME = 'text-embedding-ada-002'
    LLM_MODEL_NAME = 'gpt-4-1106-preview'
    PINECONE_INDEX_NAME = 'llm-recommender-system'
    USER_ID = '1'

    def __init__(self,
                 model_name:str = MODEL_NAME,
                 llm_model_name:str = LLM_MODEL_NAME,
                 pinecone_index_name:str = PINECONE_INDEX_NAME,
                 user_id:str = USER_ID):
        
        self.model_name = model_name
        self.llm_model_name = llm_model_name
        self.pinecone_index_name = pinecone_index_name
        self.user_id = user_id

        self.env_vars = self.__load_environment_variables()
        self.embed = self.__initialize_embedding_model()
        self.index = self.__initialize_vector_database()

        self.vectorstore = self.__initialize_vectorstore()
        self.llm = self.__initialize_llm()
        self.conversational_memory = self.__initialize_memory()

        self.tools = self.__initialize_tools()
        self.agent = self.create_agent()

    def __load_environment_variables(self):
        PINECONE_API = os.getenv("PINECONE_API")
        PINECONE_ENV = os.getenv("PINECONE_ENV")
        print(PINECONE_API)
        print(PINECONE_ENV)

        return {
            "PINECONE_API": PINECONE_API,
            "PINECONE_ENV": PINECONE_ENV
        }

    def __initialize_embedding_model(self):
        embed = OpenAIEmbeddings(
            model=self.model_name
        )
        return embed

    def __initialize_vector_database(self):
        pinecone.init(
            api_key=self.env_vars["PINECONE_API"],
            environment=self.env_vars["PINECONE_ENV"]
        )
        index = pinecone.Index(self.pinecone_index_name)
        return index
    

    def __initialize_vectorstore(self):
        text_field = "description"
        vectorstore = PineconeModified(
            self.index, self.embed.embed_query, text_field
        )
        return vectorstore
    
    def __initialize_llm(self):
        llm = ChatOpenAI(
            model_name=self.llm_model_name,
            temperature=0.0
        )
        return llm 

    def __initialize_memory(self):
        conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )
        return conversational_memory

    def __initialize_tools(self):

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
        
        ## 1. Generic Recommendation

        generic_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={'k' : 10, 
                            'filter': {'user_id' : self.user_id, 'category': 'recommended'}})
        )

        ## 2. Popular Recommendation
        pinecone_retriever = self.vectorstore.as_retriever(
                        search_kwargs={'k' : 10, 
                                        'filter': {'category': 'popular'}})

        popular_qa = pinecone_retriever | format_docs

        ## 3. Specific Recommendation
        recommended_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="map_rerank",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={'k' : 10, 
                            'filter': {'user_id' : self.user_id, 'category': 'recommended'}})
        )

        # If yes
        recommended_chain = (
            {"recommended_books" : RunnablePassthrough()}
            | PromptTemplate.from_template(
                """
                You are an expert in recommended books. \
                Give the user book recommendation books using below information. \
                Always start with "I have some books recommendation that is tailored to your taste. \
                
                Recommended Books: 
                {recommended_books}
                """
            )
            | self.llm
        )

        # If no
        popular_chain = (
            {"recommended_books": popular_qa, "question": RunnablePassthrough()}
            | PromptTemplate.from_template(
                """
                You are an expert in recommended books. \
                Give the user book recommendation books using below information. \
                Always start with "I have some popular books that I can recommend for you. \
                
                Recommended Books: 
                {recommended_books}
                """
            )
            | self.llm
        )

        full_chain = (
            {
                "topic": (
                    {"recommended_books": recommended_qa , "query": RunnablePassthrough()}
                    | ChatPromptTemplate.from_template(
                        """
                        Check if the document recommends a book. Say "yes" or "no".

                        Recommended Books: 
                        {recommended_books}

                        Classification:"""
                    )
                    | self.llm
                    | StrOutputParser()
                    ), 
                "query": RunnablePassthrough()
            }
            | RunnableBranch(
                (lambda x: "yes" in x["topic"].lower() or "Yes" in x["topic"].lower(), (lambda x :  x['query']) | recommended_qa | recommended_chain),
                (lambda x: "no" in x["topic"].lower() or "No" in x["topic"].lower(), (lambda x :  x['query']) | popular_chain),
                (lambda x :  x['query']) | popular_chain
                )
            | StrOutputParser()
        )

        tools = [
            Tool(
                name='Generic Recommendation',
                func=generic_qa.invoke,
                description=(
                    'use this tool when the user asking for book recommendation without any specific preference'
                )
            ),
            Tool(
                name='Specific Recommendation',
                func=full_chain.invoke,
                description=(
                    'use this tool when the user asking for book recommendation with a specific preference (genre, theme, etc.)'
                )
            ),
            Tool(
                name='Popular Recommendation',
                func=popular_qa.invoke,
                description=(
                    'use this tool when the user asking for popular book recommendation without any specific preference'
                )
            ),
            Tool(
                name='Generic Prompt',
                func=self.llm.invoke,
                description=(
                    'use this tool when the user asking or talk about general question'
                )
            ),
        ]

        return tools

    def create_agent(self):
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=self.conversational_memory
        )   
        return agent

    def run(self):
        # Main loop to run the agent
        while True:
            user_input = input("User: ")
            print(self.agent(user_input))

# Main Execution
if __name__ == "__main__":
    recommendation_agent = RAG()
    recommendation_agent.run()