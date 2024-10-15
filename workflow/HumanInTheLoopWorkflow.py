import asyncio
from llama_index.core.workflow import Workflow
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class HumanInTheLoopWorkflow(Workflow):
    async def run(self, *args, **kwargs):
        self.loop = asyncio.get_running_loop()
        try:
                result = await super().run(*args, **kwargs)
                logger.debug(f"{self.__class__.__name__}: HITL workflow succeeded")
                return result
        
        except Exception as e:
                logger.error(f"Workflow failed with exception: {e}")
                raise 
        finally:
            
            logger.debug(f"{self.__class__.__name__}: MLflow run ended")

