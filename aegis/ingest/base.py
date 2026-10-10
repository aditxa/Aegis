from __future__ import annotations

from abc import ABC, abstractmethod
from aegis.schemas import TextItem


class SourceConnector(ABC):
    @abstractmethod
    def fetch(self, query: str, limit: int = 25) -> list[TextItem]:
        raise NotImplementedError
