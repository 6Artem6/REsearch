"""Local architectural analysis engine (LangGraph + Ollama)."""

import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# LangGraph → jsonplus → Reviver() без allowed_objects (langchain-core 0.3.85+)
warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
    message=r"The default value of `allowed_objects`.*",
)

__version__ = "0.2.0"
