from collections.abc import Generator
from typing import Any


from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import LLMModelConfig
from dify_plugin.entities.model.message import SystemPromptMessage, UserPromptMessage
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.interfaces.agent import AgentStrategy, AgentModelConfig, ToolEntity
from pydantic import BaseModel


# 导入 logging 和自定义处理器
import logging
from dify_plugin.config.logger_format import plugin_logger_handler

# 使用自定义处理器设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)


class FunctionCallingParams(BaseModel):
    query: str
    instruction: str | None
    model: AgentModelConfig
    tools: list[ToolEntity] | None
    maximum_iterations: int = 3

class TraceAgentStrategy(AgentStrategy):
    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage]:
        round_log = self.create_log_message(
            "start_function_calling",
            {},
            ToolInvokeMessage.LogMessage.LogStatus.START,
            metadata={}
        )
        yield round_log
        fc_params = FunctionCallingParams(**parameters)
        stream = (
            ModelFeature.STREAM_TOOL_CALL in fc_params.model.entity.features
            if fc_params.model.entity and fc_params.model.entity.features
            else False
        )
        stop = (
            fc_params.model.completion_params.get("stop", [])
            if fc_params.model.completion_params
            else []
        )
        model = fc_params.model
        model_config = LLMModelConfig(**model.model_dump(mode="json"))
        prompt_messages_tools = self._init_prompt_tools(fc_params.tools)
        prompt_messages = [
            SystemPromptMessage(content=fc_params.instruction)
            **fc_params.model.history_prompt_messages,
            UserPromptMessage(content=fc_params.query)
        ]

        chunks = self.session.model.llm.invoke(
            model_config=model_config,
            prompt_messages=prompt_messages,
            stop=stop,
            stream=stream,
            tools=prompt_messages_tools,
        )
        if isinstance(chunks, Generator):
            for chunk in chunks:
                logger.info(chunk)
                yield chunk
        else:
            logger.info(chunks)
            yield chunks
        yield self.finish_log_message(
            round_log
        )

