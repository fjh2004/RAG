from typing import List, Dict
import os
from dotenv import load_dotenv
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class LLMManager:
    """
    大模型调用管理类，支持调用OpenAI API和本地开源模型。
    """
    def __init__(self, model_type: str = None, device: str = None, **generate_params):
        """
        初始化方法，区分模型类型，加载对应tokenizer和model。

        Args:
            model_type (str): 模型类型，'openai' 或 'local'。默认为环境变量中的MODEL_TYPE。
            device (str): 运行设备，'cuda' 或 'cpu'。默认为chroma_manager中的device。
            **generate_params: 生成参数，如temperature、max_new_tokens等。
        """
        self.model_type = model_type or os.getenv('MODEL_TYPE', 'openai')
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.generate_params = generate_params
        self.tokenizer = None
        self.model = None

        try:
            if self.model_type == 'openai':
                openai.api_key = os.getenv('OPENAI_API_KEY')
            elif self.model_type == 'local':
                model_path = os.getenv('LOCAL_MODEL_PATH')
                if not model_path:
                    raise ValueError('LOCAL_MODEL_PATH 未在环境变量中设置')
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
            else:
                raise ValueError(f'不支持的模型类型: {self.model_type}')
        except Exception as e:
            logger.error(f'模型初始化失败: {str(e)}')
            raise

    def generate_response(self, query: str, context_chunks: List[Dict], prompt_template: str) -> str:
        """
        接收用户问题、检索到的上下文片段、提示词模板，返回生成的回答。

        Args:
            query (str): 用户问题。
            context_chunks (List[Dict]): 检索到的上下文片段。
            prompt_template (str): 提示词模板。

        Returns:
            str: 生成的回答。
        """
        context = '\n'.join([chunk.get('text', '') for chunk in context_chunks])
        prompt = prompt_template.format(context=context, query=query)

        try:
            if self.model_type == 'openai':
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[{'role': 'user', 'content': prompt}],
                    **self.generate_params
                )
                return response['choices'][0]['message']['content'].strip()
            elif self.model_type == 'local':
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **self.generate_params)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f'模型调用失败: {str(e)}')
            raise