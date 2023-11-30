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

from RL_class import RecommendationSystemRL

import os

class RAG:
    LLM_MODEL_NAME = 'gpt-4-1106-preview'
    PINECONE_INDEX_NAME = 'llm-recommender-system'
    USER_ID = '1'

    def __init__(self,
                RecommendationSystem: RecommendationSystemRL,
                llm_model_name:str = LLM_MODEL_NAME,
                pinecone_index_name:str = PINECONE_INDEX_NAME,
                user_id:str = USER_ID,
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

        self.recomm

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
            return "\n\n".join([f"{i+1}. {d.metadata['title']} : {d.page_content}" for i,d in enumerate(docs)])
        
        def format_docs_title_source(docs):
            res = {
                'result' : "\n\n".join([f"{i+1}. {d.metadata['title']} : {d.page_content}" for i,d in enumerate(docs)]),
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

        def update_env(book_ids: list):
            # print book_ids to a file
            newrec = RecommendationSystemRL.get_new_recommendation(self.user_id, book_ids[0])
            newrec = 
            upsert new recomenndation ke vector db

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
            update_env(book_ids)

            ai_message = self.llm.invoke(message)
            return ai_message.content
        
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
            Tool(
                name='Popular Recommendation',
                func=popular_qa.invoke,
                description=(
                    'use this tool when the user asking for popular book recommendation without any specific preference'

                ),
                return_direct = True
            ),
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
        user_output = "Here are some book recommendations for you: \n" + output['result'] + "\n\n Which one do you want to read first?"
        book_ids = output['source_documents']
        return user_output, book_ids

    def run(self):
        # Main loop to run the agent
        while True:
            user_input = input("User: ")
            print(self.agent_invoke(user_input))

# Main Execution
if __name__ == "__main__":
    recommendation_agent = RAG()
    recommendation_agent.run()