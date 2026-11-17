import pytest

from assistant.config import Settings
from assistant.core import Assistant


@pytest.fixture
def assistant() -> Assistant:
    """A fresh assistant over the sample knowledge base, offline provider, in-memory stores."""
    return Assistant(Settings())


class FakeLLM:
    """Stands in for a model provider. `reply` is returned for answer prompts."""

    name = "fake"
    model = "fake-1"

    def __init__(self, reply: str = "", route: str = '{"intent": "documents"}') -> None:
        self.reply, self.route, self.calls = reply, route, []

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        self.calls.append((system, user, json_mode))
        return self.route if json_mode else self.reply
