"""Centralized Langfuse setup.

Every module that needs tracing imports from here instead of constructing
its own Langfuse client/handler, so credentials and host configuration
live in one place.
"""

import os

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

load_dotenv()


def get_langfuse_client() -> Langfuse:
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )


def get_callback_handler() -> CallbackHandler:
    return CallbackHandler()
