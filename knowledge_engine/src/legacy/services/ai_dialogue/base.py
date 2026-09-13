"""Абстракции диалога с внешними ИИ."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class DialogueTurn:
    role: str
    content: str


class BaseAIDialogueSession(ABC):
    """Многошаговый диалог в одной сессии (браузер или API)."""

    def __init__(self) -> None:
        self.history: List[DialogueTurn] = []

    @abstractmethod
    def send(self, message: str) -> str:
        """Отправить сообщение и получить ответ ассистента."""

    @abstractmethod
    def extract_reference_urls(self, text: str) -> List[str]:
        """Извлечь прямые ссылки на первоисточники из ответа."""

    def as_chat_dicts(self) -> List[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.history]
