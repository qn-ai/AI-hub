1 file changed
+226
-0
chat_terminal.py
+226
-0

import logging
import os
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Iterable, Tuple
import streamlit as st
from agrimind.util.log_config import setup_logging
from util import s3  # type: ignore  # External utility module expected in runtime environment.
from util.chat_helpers import (  # type: ignore
    process_uploaded_files_with_simple_reader_terminal,
    process_local_files_for_embedding,
    get_bedrock_response,
    log_tokens,
    build_nova_prompt_from_index,
    generate_word_doc,
    setup_titan_embedding,  # Assuming helper exposes embedding setup.
)
from llama_index.core import Settings
# === Terminal Styling ===
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
GREEN = "\033[92m"
RESET = "\033[0m"
LOGGER_NAME = "holly.chat_terminal"
def _ensure_logging() -> logging.Logger:
    """Initialize logging once per session and return the chat terminal logger."""
    if not st.session_state.get("logging_initialized"):
        setup_logging("holly")
        st.session_state["logging_initialized"] = True
    logger = setup_logging(name=LOGGER_NAME)
    return logger
logger = _ensure_logging()
logger.info("Chat terminal page initialized.")
# === Auth Check ===
if not st.session_state.get("auth"):
    logger.warning("Unauthorized access attempt detected; prompting login.")
    st.warning("You must be logged in to access this page.")
    st.stop()
st.title("Protected Page")
st.write("This content is only visible to authenticated users.")
# === Titan Embedding Setup ===
try:
    setup_titan_embedding()
    embedding_model_name = Settings.embed_model.model_name.split(":")[1].capitalize()
    logger.info("Embedding model configured: %s", embedding_model_name)
except Exception:  # pragma: no cover - defensive logging in Streamlit runtime
    logger.exception("Failed to configure Titan embedding model.")
    st.error("Unable to configure embedding model. Please check logs.")
    st.stop()
# === System Prompt ===
system_prompt = (
    "You are a friendly - but not overly friendly well mannered and helpful AI assistant to help with general queries "
    "and will provide answers based on the context provided. "
    "If you do not know the answer you will state that you do not know. "
    "Your answers should be detailed, do not cut back if you want to provide detail to respond to a user."
)
# === Session State ===
messages = []
index = None
uploaded_filenames = []
# === Pretty Print ===
def pretty_print_response(text: str, header: str | None = None, bullet_points: bool = False, width: int = 200) -> None:
    if header:
        print("-" * width)
        print(f"{BOLD}{UNDERLINE}{header.center(width)}{RESET}")
        print("-" * width)
    paragraphs = text.strip().split("\n\n")
    for para in paragraphs:
        wrapped_lines = textwrap.wrap(para.strip(), width=width)
        if bullet_points and wrapped_lines:
            print(f"{GREEN}{BOLD}• {wrapped_lines[0]}{RESET}")
            for line in wrapped_lines[1:]:
                print(f"{GREEN}{BOLD}  {line}{RESET}")
        else:
            for line in wrapped_lines:
                print(f"{GREEN}{BOLD}{line}{RESET}")
        print()
# === File Upload ===
def upload_files() -> Tuple[list[BytesIO], list[str]]:
    print("Enter file paths to upload (comma-separated):")
    file_paths = input("> ").split(",")
    file_paths = [path.strip() for path in file_paths if path.strip()]
    logger.info("User provided %d file path(s) for upload.", len(file_paths))
    files: list[BytesIO] = []
    filenames: list[str] = []
    for path in file_paths:
        try:
            with open(path, "rb") as f:
                content = f.read()
            files.append(BytesIO(content))
            filename = os.path.basename(path)
            filenames.append(filename)
            logger.debug("Loaded file for upload: %s (%d bytes)", filename, len(content))
        except Exception:
            logger.exception("Failed to read %s", path)
    logger.info("Successfully prepared %d file(s) for upload.", len(files))
    return files, filenames
def _log_index_status(source: str, file_count: int) -> None:
    """Log the status of index creation for visibility."""
    logger.info("Index updated from %s source with %d document(s).", source, file_count)
# === Main Chat Loop ===
def main() -> None:
    global index
    print("Welcome to the NDIA Terminal Chat Assistant!")
    print("Type 'load' to load pre-made documents, 'upload' to upload files, 'reset' to clear chat, or 'exit' to quit.\n")
    logger.info("Presented terminal instructions to user.")
    while True:
        user_input = input("You: ")
        logger.info("User input received: %s", user_input)
        if user_input.lower() == "exit":
            logger.info("User exited the chat terminal.")
            break
        if user_input.lower() == "upload":
            files, uploaded_filenames = upload_files()
            if files:
                print("Processing and embedding documents...")
                index = process_uploaded_files_with_simple_reader_terminal(files)
                _log_index_status("uploaded files", len(files))
                logger.debug("Uploaded filenames: %s", uploaded_filenames)
                print("Documents embedded successfully.\n")
            else:
                logger.warning("Upload command invoked but no files were prepared.")
            continue
        if user_input.lower() == "load":
            logger.info("Loading documents from S3 cache.")
            s3_object = s3.S3Utils(
                bucket="holly-vector-data-store",
                local_root=Path("./"),
                local_cache_path=Path("./docs/download"),
            )
            s3_object.sync_s3_to_local_folder_download(key_prefix="", overwrite=False)
            base_dir = Path("docs/download/holly-vector-data-store")
            if not base_dir.exists():
                logger.warning("Directory %s does not exist after sync.", base_dir)
                print(f"Directory {base_dir} does not exist.\n")
                continue
            local_files = [str(p) for p in base_dir.rglob("*") if p.is_file()]
            if not local_files:
                logger.warning("No files found under %s after sync.", base_dir)
                print("No files found in the directory.\n")
                continue
            logger.info("Processing %d file(s) from local cache.", len(local_files))
            index = process_local_files_for_embedding(local_files)
            _log_index_status("S3 cache", len(local_files))
            print("Local documents processed and embedded successfully.\n")
            continue
        if user_input.lower() == "reset":
            messages.clear()
            index = None
            logger.info("Chat session reset by user.")
            print("Chat reset.\n")
            continue
        # === Build prompt and generate response ===
        if index:
            logger.debug("Building contextual prompt from index.")
            combined_prompt = build_nova_prompt_from_index(user_input, index)
            logger.info("Querying Bedrock with contextual prompt.")
            input_tokens, output_tokens, bot_response = get_bedrock_response(
                messages=[{"role": "user", "content": combined_prompt}],
                system_prompt=system_prompt,
            )
            logger.info("Bedrock response received (input_tokens=%s, output_tokens=%s).", input_tokens, output_tokens)
            log_tokens(input_tokens, output_tokens, None)
        else:
            logger.warning("No index available; responding with guidance to upload documents.")
            bot_response = "No documents loaded. Please upload or load documents first."
        # === Display bot response ===
        logger.debug("Bot response (first 120 chars): %s", bot_response[:120])
        messages.append({"role": "assistant", "content": bot_response})
        pretty_print_response(bot_response, header="Assistant Response", bullet_points=False)
        # === Save chat history ===
        save = input("Do you want to save the chat history as a Word document? (y/n): ")
        logger.info("User save choice: %s", save)
        if save.lower() == "y":
            try:
                word_doc = generate_word_doc(messages)
                with open("chat_history.docx", "wb") as f:
                    f.write(word_doc)
                logger.info("Chat history saved to chat_history.docx.")
                print("Chat history saved as chat_history.docx\n")
            except Exception:  # pragma: no cover - defensive save guard
                logger.exception("Failed to save chat history.")
                print("Failed to save chat history. Please check logs.\n")
if __name__ == "__main__":
    main()
