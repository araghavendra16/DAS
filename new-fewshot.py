import json
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, AIMessage

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
import os
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


examples = [
    {
        "question": "How did you stage growth for your business? Can you share stories of how you scaled your company?",
        "purpose": "Seeking advice"
    },
    {
        "question": "Can you tell me about a setback you experienced, and how you recovered?",
        "purpose": "Reflection on experience"
    },
    {
        "question": "Should I attend industry conferences or workshops? Which ones?",
        "purpose": "Seeking advice"
    },
    {
        "question": "What three aspects of business building do you think are most important for early-stage founders?",
        "purpose": "Seeking advice"
    },
    {
        "question": "When you first started working, did you foresee that this is where you would be?",
        "purpose": "Reflection on experience"
    },
    {
        "question": "I'm at a crossroads right now. I can position my startup as a premium service or a low-cost service. Can we talk through the pros and cons?",
        "purpose": "Decision making"
    },
    {
        "question": "As a startup founder, what are the key personal characteristics I should work on developing to become successful?",
        "purpose": "Seeking advice"
    },
    {
        "question": "How do I identify and validate my target market to ensure I'm meeting their needs effectively?",
        "purpose": "Seeking advice"
    },
    {
        "question": "What makes your company stand out from all the other businesses in your niche?",
        "purpose": "Seeking information"
    },
    {
        "question": "What is the impact of a recession on startups?",
        "purpose": "Seeking information"
    },
    {
        "question": "I'm at a crossroads right now. I can position my startup as a premium service or a low-cost service. Can we talk through the pros and cons?",
        "answer":  "Decision making"
    },
    {
        "question": "I'm considering a pivot or new product for my business. What do you think are the pros and cons of doing this?",
        "answer":  "Decision making"
    }
]

def get_question_purpose(question):
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        Chroma,
        k=1
    )

    selected_examples = example_selector.select_examples({"question": question})
    purpose = selected_examples[0]["purpose"]
    return purpose


def get_chat_prompt_by_purpose(user_purpose):
    
    json_file_path = "prompt_book.json"  

    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    return data.get(user_purpose, "Default chat prompt if user_purpose is not found")

# Initialize Streamlit and OpenAI
def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(
        page_title="Your Startup Assistant",
        page_icon="🤖 👨‍💻"
    )

# Main function
def main():
    init()
    chat = ChatOpenAI(temperature=0)

    template = """ Please act as a startup mentor. 
        {prompt_pattern}
        {history}
        {input}
    """
    p_pattern = """Please ask me two short meaningful questions to answer my following question. When you have enough information to answer my question, create an answer to my question with consideration of all information provided to you. Please do not generate an answer until I did not provide you the answer to the asked questions."""

    template = template.format(prompt_pattern=p_pattern, history="{history}", input="{input}")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=chat)
    messages = conversation.predict(input="")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=template)
        ]

    st.header("Your own startup assistant 🤖👨‍💻")

    # Initialize question_submitted if it doesn't exist
    if "question_submitted" not in st.session_state:
        st.session_state.question_submitted = False

    # Display dropdown conditionally based on question_submitted state
    if not st.session_state.question_submitted:
        with st.sidebar:
            user_input = st.text_input("Your message: ", key="input")
            user_purpose = get_question_purpose(user_input)

            if user_input:
                st.session_state.question_submitted = True

                chat_prompt = get_chat_prompt_by_purpose(user_purpose)

                if user_purpose == "Decision making":
                    chat_prompt = f"{user_input}? {chat_prompt}"
                elif user_purpose == "Reflection on experience":
                    chat_prompt = f"{user_input}? {chat_prompt}"
                elif user_purpose == "Seeking advice":
                    chat_prompt = f"{chat_prompt} {user_input}?"
                elif user_purpose == "Seeking information":
                    chat_prompt = f"{chat_prompt} {user_input}?."


                st.session_state.messages.append(SystemMessage(content=chat_prompt))

                with st.spinner("Thinking..."):
                    response = chat(st.session_state.messages)
                st.session_state.messages.append(AIMessage(content=response.content))

    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == '__main__':
    main()

