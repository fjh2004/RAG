class PromptTemplates:
    """
    提示词模板配置类，提供单轮和多轮RAG问答的提示词模板。
    """
    @staticmethod
    def single_round_rag() -> str:
        """
        单轮RAG模板，要求模型仅使用提供的context回答query，不编造信息。

        Returns:
            str: 单轮RAG提示词模板。
        """
        return "以下是与问题相关的上下文信息：\n{context}\n\n请根据上述上下文信息回答问题：{query}。如果上下文信息不足或无关，请明确说明无法从给定信息中得出答案。"

    @staticmethod
    def multi_round_rag() -> str:
        """
        多轮对话RAG模板，结合history对话记录，保持语境连贯性，优先使用context信息。

        Returns:
            str: 多轮对话RAG提示词模板。
        """
        return "以下是对话历史记录：\n{history}\n\n以下是与当前问题相关的上下文信息：\n{context}\n\n当前问题是：{query}。请结合对话历史和上下文信息回答问题，优先使用上下文提供的信息。"

    @staticmethod
    def kg_rag_template() -> str:
        return (
            "知识图谱实体关系：\n{kg_entities}\n\n"
            "相关上下文：\n{context}\n\n"
            "请综合实体关系和上下文信息回答：{query}。"
            "若信息不足请明确说明知识缺口"
        )