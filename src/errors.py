"""Custom exceptions for this project.

Distinguishing these from third-party exceptions (openai.OpenAIError,
LangChain's own parsing errors, etc.) means calling code can catch
"a problem in our own logic" separately from "the LLM/embedding
provider had a problem" — the two usually call for different handling.
"""


class SupportDeskError(Exception):
    """Base class for all errors raised by this project's own code."""


class InvalidQuestionError(SupportDeskError):
    """The caller supplied an empty, non-string, or otherwise unusable question."""


class ClassificationError(SupportDeskError):
    """The intent classifier could not produce a valid department, even after retrying."""
