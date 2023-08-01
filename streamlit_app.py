# Conversational Retrieval QA Chatbot, built using Langflow and Streamlit
# Author: Gary A. Stafford
# Date: 2023-07-31
# Usage: streamlit run streamlit_app.py
# Requirements: pip install streamlit streamlit_chat -Uq

import logging
import sys
from typing import Optional

import requests
import streamlit as st
from streamlit_chat import message

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, stream=sys.stdout, level=logging.INFO)

BASE_API_URL = "http://localhost:7860/api/v1/process"
FLOW_ID = "cd2c3ce5-0488-4f4e-86b1-2f3c92216fc5"
# You can tweak the flow by adding a tweaks dictionary
# e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
TWEAKS = {
    "Chroma-ONPt3": {},
    "ConversationalRetrievalChain-dQ4f5": {},
    "ConversationBufferMemory-NbiVC": {},
    "HuggingFaceEmbeddings-xhPRo": {},
    "ChatOpenAI-UERtz": {},
}
BASE_AVATAR_URL = (
    "https://raw.githubusercontent.com/garystafford-aws/static-assets/main/static"
)


def main():
    st.set_page_config(page_title="Virtual Sommelier")

    st.markdown("##### Welcome to the Virtual Sommelier")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    if prompt := st.chat_input("I'm your virtual Sommelier, how may I help you?"):
        # Add user message to chat history
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "avatar": f"{BASE_AVATAR_URL}/people-64px.png",
            }
        )
        # Display user message in chat message container
        with st.chat_message(
            "user",
            avatar=f"{BASE_AVATAR_URL}/people-64px.png",
        ):
            st.write(prompt)

        # Display assistant response in chat message container
        with st.chat_message(
            "assistant",
            avatar=f"{BASE_AVATAR_URL}/sommelier-64px.png",
        ):
            message_placeholder = st.empty()
            with st.spinner(text="Thinking..."):
                assistant_response = generate_response(prompt)
                message_placeholder.write(assistant_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "avatar": f"{BASE_AVATAR_URL}/sommelier-64px.png",
            }
        )


def run_flow(inputs: dict, flow_id: str, tweaks: Optional[dict] = None) -> dict:
    api_url = f"{BASE_API_URL}/{flow_id}"

    payload = {"inputs": inputs}

    if tweaks:
        payload["tweaks"] = tweaks

    response = requests.post(api_url, json=payload)
    return response.json()


def generate_response(prompt):
    logging.info(f"question: {prompt}")
    inputs = {"question": prompt}
    response = run_flow(inputs, flow_id=FLOW_ID, tweaks=TWEAKS)
    try:
        logging.info(f"answer: {response['result']['answer']}")
        return response["result"]["answer"]
    except Exception as exc:
        logging.error(f"error: {response}")
        return "Sorry, there was a problem finding an answer for you."


if __name__ == "__main__":
    main()
