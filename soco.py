import streamlit as st
import os
from langchain.prompts import PromptTemplate
import requests
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.tools import Tool
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=api_key, model_name='llama3-8b-8192')
parser = StrOutputParser()
st.set_page_config(
    page_title="SOCO",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize chat_memory if it doesn't exist
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history")

def get_weather(city: str):
    API_KEY = os.getenv('weather_api')  # Your WeatherAPI key
    url = f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        return f"The temperature in {city} is {weather_data['current']['temp_c']}°C with {weather_data['current']['condition']['text']}."
    else:
        return f"Couldn't retrieve weather data for {city}."

# Define external tools
weather_tool = Tool(
    name="WeatherTool",
    func=get_weather,
    description="Use this tool to fetch real-time weather updates for a specific city."
)

tools = [
    Tool(
        name="GroqChat",
        func=llm.invoke,
        description="Use this tool to generate abstract and brief responses for user queries."
    ),
    weather_tool,
   
]

custom_prompt=PromptTemplate(input_variables=["user_query"],
    template='''You are a conversational agent. Respond thoughtfully: {user_query}. 
    Also stick to user word limit if mentioned.
    If unable to retive weather or conversion data, admit and give reasoning''')

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=st.session_state.chat_memory,
    prompt=custom_prompt,
    verbose=True
)

# Streamlit UI
st.title("SOCO Agent")
user_query = st.text_input("Ask me anything:")
if st.button("Submit") and user_query:
    if api_key:
        with st.spinner("Thinking..."):
            try:
                response = agent.invoke({"input": user_query})
                st.success("Response:")
                parsed_output = parser.invoke(response["output"])
                st.write(parsed_output)
            except Exception as e:
                st.exception(e)
    else:
        st.warning("Please ensure the API key is configured.")

# Display conversation history
if st.session_state.chat_memory.chat_memory.messages:
    st.subheader("Conversation History")
    for msg in st.session_state.chat_memory.chat_memory.messages:
        role = "🗣️ You:" if msg.type == "human" else "🤖 AI:"
        st.write(f"{role} {msg.content}")
