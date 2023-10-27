import json
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
import re
import nltk
import spacy
from nltk.tokenize import word_tokenize
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, AIMessage ,HumanMessage
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from streamlit_chat import message
import datetime
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
nltk.download('punkt')
examples = [
    {
        "question": "How can a startup founder stay positive and keep moving when nothing seems to be working?",
        "purpose": "Seeking advice"
    },
    {
        "question": "I have this business idea I just can't get out of my mind but I'm scared of failing, how can I get past this mental block?",
        "purpose": "Seeking advice"
    },
    {
        "question": "How do you suggest that I measure progress?",
        "purpose": "Seeking advice"
    },
    {
        "question": "What are my strengths as an entrepreneur and what should I focus on improving?",
        "purpose": "Seeking advice"
    },
    {
        "question": "What are your tips for managing and overcoming burnout?",
        "purpose": "Seeking advice"
    },
    {
        "question": "How big is the market of fitness for buniess people in Italy?",
        "purpose": "Seeking information"
    },
    {
        "question": "What do investors look for?",
        "purpose": "Seeking information"
    },
    {
        "question": "I am living in South Italy. Is there anyone in the startup networks there who could help me with getting an enterprise level customer for my startup?",
        "purpose": "Seeking information"
    },
    {
        "question": "What online marketing techniques software startups use typically?",
        "purpose": "Seeking information"
    },
    {
        "question": "Can you tell me about a setback some startups experienced, and how they recovered from it?",
        "purpose": "Seeking information"
    },
    {
        "question": "Why did our startup fail to obtain the seed fund from the angel investor?",
        "purpose":  "Reflecting on own experience"
    },

    {
        "question": "Our founding team just broke up. I wonder what went wrong with our team?",
        "purpose":  "Reflecting on own experience"
    },
     {
        "question": "How come our team missed the launch date and failed to release our MVP?",
        "purpose":  "Reflecting on own experience"
    },
     {
        "question": "We just convinced an investor to invest on our startup idea. I wonder what I did led to this positive outcome?",
        "purpose":  "Reflecting on own experience"
    },
     {
        "question": "I had a big argument with my co-founder, and I feel that I do not trust her as much as I used to do. I wonder why things happend in this way?",
        "purpose":  "Reflecting on own experience"
    },
    {
        "question": "I'm considering a pivot or new product for my business. What do you think are the pros and cons of doing this?",
        "purpose":  "Making decision"
    },
       {
        "question": "I'm at a crossroads right now. I can position my startup as a premium service or a low-cost service. Can we talk through the pros and cons?",
        "purpose":  "Making decision"
    },
       {
        "question": "I have to decide whether to accept an investment from an angel investor. I wonder what I may lose by accepting it?",
        "purpose":  "Making decision"
    },
       {
        "question": "Should I start fundraising now, and how much money should I aim to raise?",
        "purpose":  "Making decision"
    },
       {
        "question": "Our team are undecided between the B2B and B2C models. Which is the right one for our startup business?",
        "purpose":  "Making decision"
    }
]

def save_to_log(role, content):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('conversation_log.txt', 'a') as log_file:
        log_file.write(f"{timestamp} - {role}: {content}\n")

def get_question_purpose(question):
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        Chroma,
        k=1
    )

    selected_examples = example_selector.select_examples({"question": question})
    purpose = selected_examples[0]["purpose"]
    print(purpose)
    return purpose


def get_chat_prompt_by_purpose(user_purpose):
    json_file_path = "prompt_book.json"  
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    return data.get(user_purpose, "Default chat prompt if user_purpose is not found")

nlp = spacy.load("en_core_web_sm")

def is_question(input_string):
    
    input_string=str(input_string)
    if input_string.strip().endswith('?'):
        return True
    else:
        return False
    # Process the input string with spaCy
    doc = nlp(input_string)
    
    # Check if the sentence contains a question structure
    for token in doc:
        if token.dep_ == "ROOT" and token.tag_ == "VBP":
            return True  # A verb (e.g., "do") in root position indicates a question
    
    return False



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
    chat = ChatOpenAI(temperature=0, model_name = "gpt-4") # +model_name

    template = """
        {prompt_pattern}
        {history}
        {input}
    """
    p_pattern = """Please act as a startup mentor"""

    template = template.format(prompt_pattern=p_pattern, history="{history}", input="{input}")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=
        memory, prompt=prompt, llm=chat)
    messages = conversation.predict(input="")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="Act as a Startup mentor")
        ]
        st.session_state.conversation_history = [
            SystemMessage(content="Act as a startup mentor")
        ]

    st.header("Your own startup assistant 🤖👨‍💻")

    # Initialize question_submitted if it doesn't exist
    if "question_submitted" not in st.session_state:
        st.session_state.question_submitted = False

    # Display dropdown conditionally based on question_submitted state
    if not st.session_state.question_submitted:
        # with st.sidebar: 
        user_input = st.chat_input("Ask something: ", key="input")
        temp=user_input
        print(temp)
        question= is_question(temp)
        print(question)
        if user_input:
            if question:
                #print("asdddddddddddddddd")
                print(user_input)
                user_purpose = get_question_purpose(user_input)
                chat_prompt = get_chat_prompt_by_purpose(user_purpose)
                if user_purpose == "Making decision":
                    chat_prompt = f"{user_input}? {chat_prompt}"
                    st.session_state.messages.append(HumanMessage(content=chat_prompt))
                elif user_purpose == "Reflecting on own experience":
                    chat_prompt = f"{chat_prompt} {user_input}?"
                    st.session_state.messages.append(HumanMessage(content=chat_prompt))
                elif user_purpose == "Seeking advice": 
                    chat_prompt = f"{user_input}? {chat_prompt}"
                    st.session_state.messages.append(HumanMessage(content=chat_prompt))
                elif user_purpose == "Seeking information":
                    chat_prompt = f"{chat_prompt} {user_input}?"
                    st.session_state.messages.append(HumanMessage(content=chat_prompt))
                
                st.session_state.conversation_history.append(HumanMessage(content=user_input))
            #st.session_state.messages.append(SystemMessage(content=chat_prompt))
            else:
                
                st.session_state.messages.append(HumanMessage(content=user_input))
                st.session_state.conversation_history.append(HumanMessage(content=user_input))                
            print(st.session_state.messages)
            with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))
            st.session_state.conversation_history.append(AIMessage(content=response.content))
            user_input = ""

        
            
            #st.session_state.question_submitted = False  #-> to not close the conversation 

    messages = st.session_state.get('conversation_history', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
            #save_to_log("User", msg.content)
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')
            #save_to_log("Assistant", msg.content)

    if st.button("Exit"):
        with open('conversation_log1.txt', 'w') as txt_file:
            txt_file.write(f"The coversation with added prompts is: \n")
            for msg in st.session_state.messages:
                    txt_file.write(f"{msg.content}\n")

            txt_file.write(f"\n\n\n\nThe original conversation is: \n")
            for msg in st.session_state.conversation_history:
                    txt_file.write(f"{msg.content}\n")
        # Clear the messages to reset the conversation
        st.session_state.messages = []  
        st.session_state.conversation_history = []
        st.session_state.question_submitted = False  

if __name__ == '__main__':
    main()

