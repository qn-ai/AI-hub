import streamlit as st
import boto3
import os
import json
import docx
import logging
from email import import policy
from email.parser import BytesParser
from io import BytesIO

from botocore.exceptions import ClientError
from llama_index.core import Document, Settings, SimpleDirectoryReader, ServiceContext
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.bedrock import BedrockEmbedding

from typing import List, Dict, Optional, Tuple

from util.chat_helpers import process_uploaded_files_with_simple_reader, get_bedrock_response, \
    generate_word_doc, log_tokens, build_nova_prompt
from agrimind.util.log_config import setup_logging


def _reset_chat_state() -> None:
    """Clear chat-specific session state and flag confirmation."""
    logger = logging.getLogger(__name__)
    st.session_state["messages"] = []
    st.session_state["index"] = None
    st.session_state["embeddings"] = []
    st.session_state["input_tokens"] = None
    st.session_state["uploaded_files"] = []
    st.session_state["reset_chat_confirm"] = True
    logger.info("Chat state reset for user %s.", st.session_state.get("username"))

# ================================================
# Settings
# ================================================
Settings.llm = BedrockLLM(model="amazon.nova-pro-v1:0")
# embed_model_id = BedrockEmbedding(model="amazon.titan-embed-text-v2:0")
system_prompt = """
You are a friendly but not overly friendly well-mannered 
"""

Settings.embed_model = BedrockEmbedding(model_name="amazon.titan-embed-text-v2:0", region="ap-southeast-2")

# ================================================
# Auth check
# ================================================
if "auth" not in st.session_state:
    st.session_state["auth"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

username = st.session_state.get("username")
if not st.session_state.get("auth") or not username or username == "guest":
    st.warning("Please log in to access this page.")
    st.stop()

# ================================================
# Page Content
# ================================================
st.title("Protected Page")
st.write("This content is only visible to authenticated users.")

# ================================================
# Model and Logging
# ================================================
embedding_model_name = Settings.embed_model.model_name.split(":")[1].capitalize()

if "logging_initialized" not in st.session_state:
    logger = setup_logging(__name__)
    st.session_state["logging_initialized"] = True
else:
    logger = logging.getLogger(__name__)
logger.info("chat_app loaded.")

# ================================================
# Session State Init
# ================================================
for key, default in {
    "messages": [],
    "uploaded_files": [],
    "embeddings": [],
    "index": None,
    "input_tokens": None
}.items():
    st.session_state.setdefault(key, default)

if st.session_state.pop("reset_chat_confirm", False):
    st.success("Chat reset.")

# ================================================
# UI
# ================================================
st.title("How can I help today? Type in the chat bar below to start...")

st.sidebar.header("Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["txt", "docx", "msg", "pdf", "eml", "png", "gif", "jpeg", "jpg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("Processing and embedding uploaded documents...")
    index = process_uploaded_files_with_simple_reader(uploaded_files)
    st.session_state.index = index
    st.success("Documents embedded successfully.")

# ================================================
# Interpret Images with Nova
# ================================================
image_descriptions = []
for file in uploaded_files:
    if file.name.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
        st.info(f"Interpreting image: {file.name}")
        description = interpret_image_with_nova(file, prompt="Describe this image in detail.")
        image_descriptions.append((file.name, description))
        st.success(f"Image interpreted: {file.name}")

# ================================================
# Chat
# ================================================
top_nodes: List = []
user_input = st.chat_input("Ask a question")

if user_input:
    with st.chat_message("user"):
        logger.info("User message received")
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Build prompt with context if available
    if st.session_state.index is not None:
        logger.info("Index available, building contextual prompt...")
        combined_prompt, top_nodes = build_nova_prompt_from_index(user_input, st.session_state.index)
    else:
        logger.info("Index not available, using raw input.")
        combined_prompt, top_nodes = user_input, []

    # Stream response from Bedrock
    logger.info("Streaming Bedrock response...")
    full_response = ""
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        for chunk in stream_bedrock_response(
            messages=st.session_state.messages,
            system_prompt=system_prompt
        ):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    # Save to session
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ================================================
# Show Image Interpretations
# ================================================
if image_descriptions:
    for fname, desc in image_descriptions:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**Image ({fname}) interpretation:**\n{desc}"
        })
        st.markdown(f"**Image ({fname}) interpretation:**\n{desc}")

# ================================================
# Show Retrieved Context Chunks
# ================================================
if top_nodes:
    st.subheader("Retrieved Context Chunks")
    for i, node in enumerate(top_nodes, start=1):
        metadata = node.node_metadata
        label = f"Context Node {i}"
        if "page_number" in metadata:
            label += f" (Page {metadata['page_number']})"
        elif "line_number" in metadata:
            label += f" (Line {metadata['line_number']})"
        elif "paragraph_number" in metadata:
            label += f" (Paragraph {metadata['paragraph_number']})"
        with st.expander(label):
            st.markdown(node.node_text.strip())

# ================================================
# Reset Chat
# ================================================
st.button("Reset Chat", on_click=_reset_chat_state)

# ================================================
# Save Chat History
# ================================================
if st.button("Save Chat History as Word Document"):
    word_doc = generate_word_doc(st.session_state.messages)
    st.download_button("Download chat_history.docx", data=word_doc, file_name="chat_history.docx")
    logger.info("User %s saved chat session.", st.session_state.get("username", "guest"))

st.markdown("---")
