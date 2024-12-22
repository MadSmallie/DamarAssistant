import openai
import os
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PineCone_API_KEY")

def main():

    pc = Pinecone(api_key=pinecone_api_key)

    pinecone_index = pc.Index("damarvectorstore")

    vector_store = PineconeVectorStore(pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    from llama_index.core.memory import ChatMemoryBuffer

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            "You are a helpful assistant! Please answer questions solely based on the provided documents!"
        ),
    )

    chat_history = []

    def chat_function(user_input):
        """
        Handles the chat interaction.
        Takes user input, retrieves the response from the chat engine, 
        and appends the conversation to the chat history.
        """
        global chat_history
        response = chat_engine.chat(user_input)
        chat_history.append(("You", user_input))
        chat_history.append(("Assistant", response.response))
        return chat_history
    

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Damar Onboarding Training Assitant")
        
    # Display Chat History
        chat_output = gr.Chatbot(label="Conversation History")
        
    # Chat Interface
        
        with gr.Row():
            user_input = gr.Textbox(label="Questions", placeholder="Please ask any question you have about Damar and the BTP program",scale=2)
            submit_button = gr.Button("Submit",scale=0)
        # Clear History Button
        
        clear_button = gr.Button("Refresh")
        
        # Button actions
        submit_button.click(chat_function, inputs=user_input, outputs=chat_output)
        #clear_button.click(zmxsb, outputs=chat_output)
        clear_button.click(lambda: None, outputs=chat_output)
        clear_button.click(chat_engine.reset())

    demo.launch(share=True)


if __name__ == "__main__":
    main()