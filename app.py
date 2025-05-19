import streamlit as st
import time
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Jarvis AI Chatbot", page_icon="🤖")
st.markdown(
    "<h1 style='text-align: center;'>🤖 Jarvis: All About Ai</h1>",
    unsafe_allow_html=True,
)

openai_api_key = st.sidebar.text_input("🔐 Enter your OpenAI API Key:", type="password")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )


def typewriter(text, delay=0.03):
    placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(
            f"""
            <div style='text-align: left;'>
                <div style='background-color: #2ecc71; color: white; padding: 12px 15px; border-radius: 0px 15px 15px 15px;
                            display: inline-block; max-width: 80%; box-shadow: 2px 3px 8px rgba(0,0,0,0.1); 
                            text-align: left; font-size: 16px; line-height: 1.5;'>
                    🤖 {typed}<span class='cursor'>▌</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(delay)
    placeholder.markdown(
        f"""
        <div style='text-align: left;'>
            <div style='background-color: #2ecc71; color: white; padding: 12px 15px; border-radius: 0px 15px 15px 15px;
                        display: inline-block; max-width: 80%; box-shadow: 2px 3px 8px rgba(0,0,0,0.1); 
                        text-align: left; font-size: 16px; line-height: 1.5;'>
                🤖 {typed}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_response(question):
    if question.lower() in ["hi", "hello", "hey"]:
        return "Hello! I'm Jarvis. How can I help you with Artificial Intelligence today?"

    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
                    You are Jarvis, an AI assistant that only answers questions related to Artificial Intelligence.

                    Your job is to respond helpfully **only if the question is about AI**. If it's a general question like "How are you?", reply politely that you are functioning well and always ready to assist **with AI-related queries only**.

                    Examples:
                    User: How are you?
                    Jarvis: I am functioning well and always ready to assist with any AI-related queries.

                    User: What's the weather?
                    Jarvis: Sorry, I can only answer questions related to Artificial Intelligence.

                    User: What is deep learning?
                    Jarvis: Deep learning is a subset of AI that focuses on neural networks with many layers...

                    Now respond to this:
                    Question: {question}
                    """,
    )

    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory)
    return chain.run(question)


for msg in st.session_state.chat_history:
    st.markdown(
        f"""
        <div style='text-align: left; margin-bottom: 10px;'>
            <div style='background-color: #3498db; color: white; padding: 12px 15px; border-radius: 0px 15px 15px 15px; 
                        display: inline-block; max-width: 80%; box-shadow: 2px 3px 8px rgba(0,0,0,0.1); 
                        text-align: left; font-size: 16px; line-height: 1.5;'>
                👤 {msg['user']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style='text-align: left; margin-bottom: 20px;'>
            <div style='background-color: #2ecc71; color: white; padding: 12px 15px; border-radius: 0px 15px 15px 15px; 
                        display: inline-block; max-width: 80%; box-shadow: 2px 3px 8px rgba(0,0,0,0.1); 
                        text-align: left; font-size: 16px; line-height: 1.5;'>
                🤖 {msg['assistant']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if openai_api_key.startswith("sk-"):
    user_input = st.chat_input("Type your question about AI...")
    if user_input:
        st.markdown(
            f"""
            <div style='text-align: left; margin-bottom: 10px;'>
                <div style='background-color: #3498db; color: white; padding: 12px 15px; border-radius: 0px 15px 15px 15px; 
                            display: inline-block; max-width: 80%; box-shadow: 2px 3px 8px rgba(0,0,0,0.1); 
                            text-align: left; font-size: 16px; line-height: 1.5;'>
                    👤 {user_input}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("Jarvis is thinking..."):
            response = generate_response(user_input)

        typewriter(response)

        st.session_state.chat_history.append(
            {"user": user_input, "assistant": response}
        )
else:
    st.warning("Please enter a valid OpenAI API key to chat.")
