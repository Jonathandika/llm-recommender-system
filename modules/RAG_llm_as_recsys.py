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
from langchain.output_parsers import CommaSeparatedListOutputParser

from modules.helper.PineconeModified import PineconeModified
from modules.helper.PineconeSelfQueryRetriever import PineconeSelfQueryRetriever

from FlagEmbedding import FlagModel

import os

class RAG_llm:
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

        return {
            "PINECONE_API": PINECONE_API,
            "PINECONE_ENV": PINECONE_ENV
        }

    def __initialize_embedding_model(self):
        model = FlagModel('BAAI/bge-large-en-v1.5', 
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                use_fp16=True)
        embed = lambda x: model.encode(x).tolist()     
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
            self.index, self.embed, text_field
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

        def format_docs_title(docs):
            return "\n\n".join([f"{i+1}. {d.metadata['title']} : {d.page_content}" for i,d in enumerate(docs)])

        pineconeRetreiver = self.vectorstore.as_retriever(
            search_kwargs={'k' : 10}
        )

        candidate_books_qa = pineconeRetreiver | format_docs_title

        chain = (
            {"candidate_books" : candidate_books_qa, 'user_query' : RunnablePassthrough()}
            | PromptTemplate.from_template(
                """
                You are a recommender. Based on the user's profile and behaviors, user_query, you are
                recommending 5 suitable books that he/she will like. Output the title of recommended book and the reason why you recommend it.
                
                User Information: 
                Name: Vabel
                Books Read:
                0	Angels & Demons (Robert Langdon, #1)
                1	The Girl Who Played with Fire (Millennium, #2)
                2	The Psychopath Test: A Journey Through the Madness Industry
                3	The Old Man and the Sea
                4	A Wolf at the Table
                5	The Princess Bride
                6	People of the Book
                7	Nightfall
                8	A Thousand Splendid Suns
                9	Desperate Passage: The Donner Party's Perilous Journey West
                10	Look Me in the Eye: My Life with Asperger's
                11	Nickel and Dimed: On (Not) Getting by in America
                12	The Shining
                13	The Thorn Birds
                14	The Jungle
                15	The Magus
                16	The Gnostic Gospels
                17	Flowers in the Attic (Dollanganger, #1)
                18	The Martian Chronicles
                19	The Map Thief
                20	Crust and Crumb: Master Formulas for Serious Bread Bakers
                21	Mouse Guard: Winter 1152 (Mouse Guard, #2)
                22	The Tenth Muse: My Life in Food
                23	The Way We Never Were: American Families & the Nostalgia Trap
                24	The Dragonriders of Pern (Dragonriders of Pern, #1-3)
                25	The Blessing Way (Leaphorn & Chee, #1)
                26	Blow Fly (Kay Scarpetta, #12)
                27	The Honk and Holler Opening Soon
                28	Cadillac Jack
                29	Julia Child & More Company
                30	Angry Housewives Eating Bon Bons

                Candidate Books:
                {candidate_books}

                User Query: 
                {user_query}
                """
            )
            | self.llm
        )

        tools = [
            Tool(
                name='Recommendation',
                func=chain.invoke,
                description=(
                    'use this tool when the user asking for book recommendation'
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
    recommendation_agent = RAG_llm()
    recommendation_agent.run()