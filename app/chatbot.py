import streamlit as st
from haystack import Pipeline, Document
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import StreamingChunk, ChatMessage
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from tika import parser


class Chatbot:

    def __init__(self):
        self.document_store = InMemoryDocumentStore()
        self.document_store.write_documents(
            [
                Document(content=parser.from_file("../data/genai/Owners_Manual_tesla.pdf")['content']),
            ]
        )
        self.large_language_model = OllamaChatGenerator(
            model='llama3.1',
            streaming_callback=self.streamlit_write_streaming_chunk
        )

        self.messages = [ChatMessage.from_system("\nYou are a helpful, respectful and honest assistant")]

    def query(self, query: str) -> str:
        # Create a new Streamlit container for the AI's response.
        self.placeholder = st.empty()

        # Initialize an empty list for response tokens.
        self.tokens = []

        # Add the user's query to the chat history.
        self.messages.append(ChatMessage.from_user(query))

        # Send the chat history to the language model and get the response.
        response = self.large_language_model.run(self.messages)

        # Check if the response contains valid replies.
        if 'replies' in response:
            response_content = response['replies'][0].content
            # Add the assistant's response to the chat history.
            self.messages.append(ChatMessage.from_assistant(response_content))
            return response_content
        else:
            raise Exception('No valid response or unexpected response format.')

    def streamlit_write_streaming_chunk(self, chunk: StreamingChunk):
        # Append the latest streaming chunk to the tokens list.
        self.tokens.append(chunk.content)

        # Update the Streamlit container with the current stream of tokens.
        self.placeholder.write("".join(self.tokens))

    def add_message_to_chat_history(self, message: ChatMessage):
        self.messages.append(message)