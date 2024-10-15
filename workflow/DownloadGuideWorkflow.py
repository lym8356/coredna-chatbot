import logging
from typing import Optional
import uuid
from llama_index.core import Settings

from constants import CHUNK_SIZE, OPENAI_EMBEDDING_MODEL, OPENAI_MODEL
from workflow import HumanInTheLoopWorkflow
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    step,
    draw_all_possible_flows,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


Settings.chunk_size = CHUNK_SIZE
Settings.embed_model = OPENAI_EMBEDDING_MODEL
Settings.llm = OPENAI_MODEL


class DownloadGuideWorkflow(HumanInTheLoopWorkflow):
    wid: Optional[uuid.UUID] = uuid.uuid4()

    def __init__(self, wid: Optional[uuid.UUID], *args, **kwargs):
        self.wid = wid
        super().__init__(*args, **kwargs)
        class_name = self.__class__.__name__


