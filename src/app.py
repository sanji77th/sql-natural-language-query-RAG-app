import os
import dotenv
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


dotenv.load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        AIMessage(content="Ha haa!, I know your company database secrets. Ask from me!")
    ]


def init_database(host, port, user, password, database):
    database_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(database_uri)


def get_sqlchain(database):
    template = """
    Based on the database schema and the chat history provided below, i want you to write the my sql query that answer for the user question. Give the query as one sentence string.
    Do not give any other text with SQL quey. Just give the SQL query as a single sentence.
    database schema is {schema}
    chat history is {chat_history}
    question of the user is {question}
    Sql query:
    """

    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return database.get_table_info()
    
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )


def get_response(user_query, database, chat_history):
    chain_sql = get_sqlchain(database)

    print(chain_sql)

    natural_template = """
    Based on the given schema, chat history, question, sql query and the sql response, give a natural language answer.

    schema: {schema}
    chat history is: {chat_history}
    question: {question}
    sql query: {query}
    sql response: {response}
    """

    natural_prompt = ChatPromptTemplate.from_template(natural_template)

    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)

    def run_query(query):
        return database.run(query)
    
    full_chain = (RunnablePassthrough.assign(query=chain_sql).assign(
    schema=lambda _: database.get_table_info(),
    response= lambda vars: database.run(vars["query"]))
    | natural_prompt
    | llm
    | StrOutputParser()
    )

    return full_chain.invoke({
        'question':user_query,
        'chat_history': chat_history
    })





st.set_page_config(page_title="chat with mySQL", page_icon=":speech balloon:")


st.title("Let's chat about your data 💬")


user_text = st.chat_input("Type your question here...")


with st.sidebar:
    st.subheader("Connect Here")
    st.write("Hey you can give your credentials here and let me to know your data secrets..😈")

    st.text_input("Host",value="localhost", key="Host")
    st.text_input("Port",value="3306", key="Port")
    st.text_input("User",value="root", key="User")
    st.text_input("Password", type="password", value="123456", key="Password")
    st.text_input("Database",value="company_database", key="Database")

    if st.button("Connect"):
        with st.spinner("Conecting to your datacave....🗿"):
            try:
                database = init_database(
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Database"]
                )
                st.session_state["QueryDB"] = database
                st.success("Connected to the Databse!", icon="✅")
            except Exception as e:
                st.error('Wrong database credentials', icon="🚨")
    

for text in st.session_state["chat_history"]:
    if isinstance(text,AIMessage):
        with st.chat_message("AI"):
            st.markdown(text.content)
    if isinstance(text,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(text.content)


if user_text is not None and user_text.strip() != "":
    st.session_state["chat_history"].append(HumanMessage(content=user_text))

    with st.chat_message("Human"):
        st.markdown(user_text)

    chain_sql = get_sqlchain(st.session_state["QueryDB"])
    response = get_response(user_text, st.session_state["QueryDB"], st.session_state["chat_history"])
    st.session_state["chat_history"].append(AIMessage(content=response))

    with st.chat_message("AI"):
        st.markdown(response)
