import streamlit as st
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from chatbot import Chatbot

# Constants to store key names in the config dictionary
TITLE_NAME = 'title_name'
UI_RENDERED_MESSAGES = 'ui_rendered_messages'
CHAT_HISTORY = 'chat_history'
CONVERSATIONAL_PIPELINE = 'conversational_pipeline'


def main():
    config = load_config()
    initialize_session_state(config)
    setup_page()
    render_chat_history()
    manage_chat()


def load_config():
    return {
        TITLE_NAME: 'Summerschool Bot',
        UI_RENDERED_MESSAGES: [],
        CHAT_HISTORY: [],
        CONVERSATIONAL_PIPELINE: Chatbot()
    }


def setup_page():
    st.set_page_config(page_title=st.session_state[TITLE_NAME])
    st.title(st.session_state[TITLE_NAME])


def initialize_session_state(config):
    for key, value in config.items():
        if key not in st.session_state:
            st.session_state[key] = value


def manage_chat():
    if prompt := st.chat_input('What can we help you with?'):
        # Render user message.
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state[UI_RENDERED_MESSAGES].append({'role': 'user', 'content': prompt})

        # Render AI assistant's response.
        with st.chat_message('assistant'):
            with st.spinner('Generating response . . .'):
                response = st.session_state[CONVERSATIONAL_PIPELINE].query(prompt)
        st.session_state[UI_RENDERED_MESSAGES].append({'role': 'assistant', 'content': response})


def render_chat_history():
    for message in st.session_state[UI_RENDERED_MESSAGES]:
        with st.chat_message(message['role']):
            st.markdown(message['content'])


if __name__ == '__main__':
    main()