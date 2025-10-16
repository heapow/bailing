from abc import ABC, abstractmethod
import openai
import requests
import json
import logging

logger = logging.getLogger(__name__)


class LLM(ABC):
    @abstractmethod
    def response(self, dialogue):
        pass


class OpenAILLM(LLM):
    def __init__(self, config):
        self.model_name = config.get("model_name")
        self.api_key = config.get("api_key")
        self.base_url = config.get("url")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def response(self, dialogue):
        try:
            responses = self.client.chat.completions.create(  #) ChatCompletion.create(
                model=self.model_name,
                messages=dialogue,
                stream=True
            )
            for chunk in responses:
                yield chunk.choices[0].delta.content
                #yield chunk.choices[0].delta.get("content", "")
        except Exception as e:
            logger.error(f"Error in response generation: {e}")

    def response_call(self, dialogue, functions_call):
        try:
            responses = self.client.chat.completions.create(  #) ChatCompletion.create(
                model=self.model_name,
                messages=dialogue,
                stream=True,
                tools=functions_call
            )
            #print(responses)
            for chunk in responses:
                yield chunk.choices[0].delta.content, chunk.choices[0].delta.tool_calls
                #yield chunk.choices[0].delta.get("content", "")
        except Exception as e:
            logger.error(f"Error in response generation: {e}")


class OllamaLLM(LLM):
    def __init__(self, config):
        self.model_name = config.get("model_name", "qwen2.5")
        self.base_url = config.get("url", "http://localhost:11434/api/chat")

    def response(self, dialogue):
        payload = {
            "model": self.model_name,
            "messages": dialogue,
            "stream": True
        }
        try:
            resp = requests.post(self.base_url, json=payload, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode())
                content = data.get("message", {}).get("content")
                if content:
                    yield content
        except Exception as e:
            logger.error(f"OllamaLLM stream error: {e}")

    def response_call(self, dialogue, tools):
        """
        支持流式工具调用：
        tools: list of tool definitions, e.g. [{"type":"function","function":{...}}, ...]
        """
        payload = {
            "model": self.model_name,
            "messages": dialogue,
            "stream": True,
            "tools": tools
        }
        try:
            resp = requests.post(self.base_url, json=payload, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode())
                msg = data.get("message", {})
                content = msg.get("content")
                tool_calls = msg.get("tool_calls")
                yield content, tool_calls
        except Exception as e:
            logger.error(f"OllamaLLM tool-call error: {e}")


class DifyLLM(LLM):
    """
    Dify平台的LLM实现
    支持Dify的聊天应用和Agent应用
    """
    def __init__(self, config):
        self.api_key = config.get("api_key")
        self.base_url = config.get("url", "https://api.dify.ai/v1")
        self.user = config.get("user", "bailing-user")
        self.conversation_id = None  # 用于保持对话上下文
        
    def response(self, dialogue):
        """
        调用Dify的chat-messages接口进行对话
        dialogue: 标准的消息列表格式 [{"role": "user", "content": "..."}, ...]
        """
        try:
            # 提取最后一条用户消息作为query
            query = ""
            for msg in reversed(dialogue):
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
            
            if not query:
                logger.warning("No user message found in dialogue")
                return
            
            # 构建Dify API请求
            url = f"{self.base_url}/chat-messages"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": {},
                "query": query,
                "response_mode": "streaming",
                "user": self.user
            }
            
            # 如果有conversation_id，添加到请求中以保持上下文
            if self.conversation_id:
                payload["conversation_id"] = self.conversation_id
            
            # 发送流式请求
            resp = requests.post(url, headers=headers, json=payload, stream=True)
            resp.raise_for_status()
            
            # 处理流式响应
            for line in resp.iter_lines():
                if not line:
                    continue
                
                # Dify返回的是 "data: {json}" 格式
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    line_str = line_str[6:]  # 移除 "data: " 前缀
                
                try:
                    data = json.loads(line_str)
                    event = data.get("event")
                    
                    # 处理不同类型的事件
                    if event == "message":
                        # 普通消息事件，包含文本内容
                        content = data.get("answer", "")
                        if content:
                            yield content
                    elif event == "agent_message":
                        # Agent消息事件
                        content = data.get("answer", "")
                        if content:
                            yield content
                    elif event == "message_end":
                        # 消息结束事件，保存conversation_id
                        self.conversation_id = data.get("conversation_id")
                    elif event == "error":
                        # 错误事件
                        logger.error(f"Dify API error: {data}")
                        
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse line: {line_str}, error: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"DifyLLM stream error: {e}")
    
    def response_call(self, dialogue, tools):
        """
        Dify的工具调用支持
        注意：Dify的工具调用是在平台上配置的，这里主要处理响应
        """
        try:
            # 提取最后一条用户消息作为query
            query = ""
            for msg in reversed(dialogue):
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
            
            if not query:
                logger.warning("No user message found in dialogue")
                return
            
            # 构建Dify API请求
            url = f"{self.base_url}/chat-messages"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": {},
                "query": query,
                "response_mode": "streaming",
                "user": self.user
            }
            
            # 如果有conversation_id，添加到请求中
            if self.conversation_id:
                payload["conversation_id"] = self.conversation_id
            
            # 发送流式请求
            resp = requests.post(url, headers=headers, json=payload, stream=True)
            resp.raise_for_status()
            
            # 处理流式响应
            for line in resp.iter_lines():
                if not line:
                    continue
                
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    line_str = line_str[6:]
                
                try:
                    data = json.loads(line_str)
                    event = data.get("event")
                    
                    if event == "message":
                        content = data.get("answer", "")
                        yield content, None
                    elif event == "agent_message":
                        content = data.get("answer", "")
                        yield content, None
                    elif event == "agent_thought":
                        # Agent思考过程，可以记录但不返回
                        logger.debug(f"Agent thought: {data}")
                    elif event == "message_end":
                        self.conversation_id = data.get("conversation_id")
                    elif event == "error":
                        logger.error(f"Dify API error: {data}")
                        
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse line: {line_str}")
                    continue
                    
        except Exception as e:
            logger.error(f"DifyLLM tool-call error: {e}")


def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls = globals().get(class_name)
    if cls:
        # 创建并返回实例
        return cls(*args, **kwargs)
    else:
        raise ValueError(f"Class {class_name} not found")


if __name__ == "__main__":
    # 创建 DeepSeekLLM 的实例
    deepseek = create_instance("DeepSeekLLM", api_key="your_api_key", base_url="your_base_url")
    dialogue = [{"role": "user", "content": "hello"}]

    # 打印逐步生成的响应内容
    for chunk in deepseek.response(dialogue):
        print(chunk)
