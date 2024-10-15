from llama_index.core.workflow import Event
from typing import Optional


class InitializeEvent(Event):
    pass


class ConciergeEvent(Event):
    request: Optional[str]
    just_completed: Optional[str]
    need_help: Optional[bool]


class OrchestratorEvent(Event):
    request: str

class StockLookupEvent(Event):
    request: str

class AuthenticateEvent(Event):
    request: str

class AccountBalanceEvent(Event):
    request: str

class TransferMoneyEvent(Event):
    request: str

# more events here