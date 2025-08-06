from typing import List, Dict
import uuid
from datetime import datetime

class ConversationManager:
    """
    对话历史管理类，用于存储和管理用户多轮对话历史。
    使用内存字典实现，支持后续扩展为数据库。
    """
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def create_session(self) -> str:
        """
        创建新会话，返回唯一session_id。

        Returns:
            str: 新会话的唯一标识符。
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'session_id': session_id,
            'messages': []
        }
        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        添加消息到指定会话。

        Args:
            session_id (str): 会话标识符。
            role (str): 消息角色，'user' 或 'assistant'。
            content (str): 消息内容。
        """
        if session_id not in self.sessions:
            raise ValueError('会话不存在')
        self.sessions[session_id]['messages'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })

    def get_history(self, session_id: str, max_turns: int = 5) -> List[Dict]:
        """
        获取指定会话的最近N轮对话历史。

        Args:
            session_id (str): 会话标识符。
            max_turns (int): 最大返回的对话轮数，默认为5。

        Returns:
            List[Dict]: 最近N轮对话历史列表。
        """
        if session_id not in self.sessions:
            raise ValueError('会话不存在')
        return self.sessions[session_id]['messages'][-max_turns:]

    def delete_session(self, session_id: str) -> None:
        """
        删除指定会话。

        Args:
            session_id (str): 会话标识符。
        """
        if session_id not in self.sessions:
            raise ValueError('会话不存在')
        del self.sessions[session_id]