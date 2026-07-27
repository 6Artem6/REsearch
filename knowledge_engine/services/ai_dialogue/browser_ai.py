"""Legacy Perplexity session — используйте gemini_session.BrowserGeminiDialogueSession."""

from knowledge_engine.services.ai_dialogue.gemini_session import (
    BrowserGeminiDialogueSession,
)

# Совместимость импортов
BrowserAIDialogueSession = BrowserGeminiDialogueSession

__all__ = ["BrowserAIDialogueSession", "BrowserGeminiDialogueSession"]
