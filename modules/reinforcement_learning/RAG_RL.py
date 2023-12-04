from dotenv import dotenv_values
import pinecone
import pandas as pd
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
from langchain.agents import (initialize_agent, AgentType)

from langchain.output_parsers import CommaSeparatedListOutputParser

from modules.helper.PineconeModified import PineconeModified
from modules.helper.PineconeSelfQueryRetriever import PineconeSelfQueryRetriever

from FlagEmbedding import FlagModel

from modules.RecommendationSystem_RL import RecommendationSystemRL
from modules.IndexEmbeddingVectors_RL import IndexEmbeddingVectors

import pickle

import os

class RAG_RL:
    LLM_MODEL_NAME = 'gpt-4-1106-preview'
    PINECONE_INDEX_NAME = 'rl-llm-recsys'
    USER_ID = 185

    def __init__(self,
                recommendation_system: RecommendationSystemRL,
                llm_model_name:str = LLM_MODEL_NAME,
                pinecone_index_name:str = PINECONE_INDEX_NAME,
                user_id:int = USER_ID,
                ):
        
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

        self.recommendation_system = recommendation_system
        self.index_embedding_vectors = IndexEmbeddingVectors()
        
        if 'book_ids.pkl' in os.listdir('output/RL/'):
            with open('output/RL/book_ids.pkl', 'rb') as f:
                self.book_ids = pickle.load(f)


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

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])
        
        def format_docs_title(docs):
            return "\n\n".join([f"{i+1}. {d.metadata['book_title']} : {d.page_content}" for i,d in enumerate(docs)])
        
        def format_docs_title_source(docs):
            res = {
                'category' : 'specific',
                'result' : "\n\n".join([f"{i+1}. {d.metadata['book_title']}" for i,d in enumerate(docs)]),
                'source_documents' : [d.metadata['book_id'] for d in docs]
            }
            return str(res)
        
        ## 1. Generic Recommendation

        pinecone_retriever = self.vectorstore.as_retriever(
                search_kwargs={'k' : 5, 
                                'filter': {'user_id' : self.user_id,'category': 'recommended'}})

        generic_qa = pinecone_retriever | format_docs_title_source

        ## 2. Popular Recommendation
        pinecone_retriever = self.vectorstore.as_retriever(
                        search_kwargs={'k' : 5, 
                                        'filter': {'category': 'popular'}})

        popular_qa = pinecone_retriever | format_docs

        ## 3. Specific Recommendation
        pinecone_retriever = self.vectorstore.as_retriever(
                search_kwargs={'k' : 5, 
                            'filter': {'user_id' : self.user_id, 'category': 'recommended'}})

        recommended_qa = pinecone_retriever | format_docs_title
        recommended_qa_with_source =  pinecone_retriever | format_docs_title_source
    

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
                (lambda x: "yes" in x["topic"].lower() or "Yes" in x["topic"].lower(), (lambda x :  x['query']) | recommended_qa_with_source),
                (lambda x: "no" in x["topic"].lower() or "No" in x["topic"].lower(), (lambda x :  x['query']) | popular_chain),
                (lambda x :  x['query']) | popular_chain
                )
            | StrOutputParser()
        )

        ## 4. Feedback loop
        output_parser = CommaSeparatedListOutputParser()

        def update_env(chosen_books: list):
            # Map book_id
            
            book_index = int(chosen_books[0]) - 1
            book_accepted_id = self.book_ids[book_index] 

            print(f"book_accepted_id: {book_accepted_id}")
            # print book_ids to a file
            rec_init = pd.read_csv('output/RL/rec_initial.csv')
            user_id_encoded = rec_init[rec_init['user_id']==self.user_id].user_encoded.unique()[0]
            newrec = self.recommendation_system.get_new_recommendation(user_id_encoded, book_accepted_id)
            newrec.dropna(inplace=True)

            # newrec.to_csv('output/RL/rec_new.csv', index=False)

            self.index_embedding_vectors.delete_vectors(self.user_id)
            self.index_embedding_vectors.index_embedding_vectors(newrec, 'recommended')
            return

        def feedback(message):
            def parse_feedback(message):
                chain = (
                        {'user_feedback' : RunnablePassthrough()}
                        | PromptTemplate.from_template(
                            """
                            You are a parser. Based on the user's feedback output the book id of the book that the user likes. \
                            
                            User Feedback:
                            {user_feedback}

                            Your response should be a list of comma separated values, eg: `foo, bar, baz`
                            """
                        )
                        | self.llm
                        | output_parser
                )
                return chain.invoke(message)

            # hrsnya ada thread yg ngeupdate env
            book_ids = parse_feedback(message)
            print(f'Book_ids: {book_ids}')
            update_env(book_ids)

            ai_message = self.llm.invoke(message)
            
            res = {
                'category' : 'general',
                'result' : ai_message.content
            }

            return str(res)
        
        tools = [
            Tool(
                name='Generic Recommendation',
                func=generic_qa.invoke,
                description=(
                    'use this tool when the user asking for book recommendation without any specific preference (not popular) just input "book" as the query parameter'
                ),
                return_direct = True
            ),
            Tool(
                name='Specific Recommendation',
                func=full_chain.invoke,
                description=(
                    'use this tool when the user asking for book recommendation with a specific preference (genre, theme, etc.)'
                ),
                return_direct = True
            ),
            # Tool(
            #     name='Popular Recommendation',
            #     func=popular_qa.invoke,
            #     description=(
            #         'use this tool when the user asking for popular book recommendation without any specific preference'

            #     ),
            #     return_direct = True
            # ),
            Tool(
                name='Generic Prompt',
                func=feedback,
                description=(
                    'use this tool when the user give a feedback to the recommendation and input the user input as the query parameter'
                ),
                return_direct = True

            ),
        ]

        return tools

    def create_agent(self):
        agent = initialize_agent(
                agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                tools=self.tools,
                llm=self.llm,
                verbose=True,
                memory=self.conversational_memory,
                return_source_documents=True
        )
        return agent

    def agent_invoke(self, message):
        res = self.agent.invoke(message)
        output = eval(res['output'])
        print(output)
        if output['category'] == 'specific':
            user_output = "Here are some book recommendations for you: \n" + output['result'] + "\n\n Which one do you want to read first?"
            self.book_ids = output['source_documents']
            
            with open('output/RL/book_ids.pkl', 'wb') as f:
                pickle.dump(self.book_ids, f)

            return user_output
        else:
            return output['result']

    def run(self):
        # Main loop to run the agent
        while True:
            user_input = input("User: ")
            print(self.agent_invoke(user_input))

# Main Execution
if __name__ == "__main__":
    config = dotenv_values(".env")

    rs = RecommendationSystemRL(retrain=False)
    recommendation_agent = RAG_RL(rs)
    recommendation_agent.run()