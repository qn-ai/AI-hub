def retrieve_and_generate(
        self,
        question: str,
        prompt_template: str = GENERATION_PROMPT_TEMPLATE,
    ) -> dict:
        try:
            return self.client.retrieve_and_generate(
                input={"text": question},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.knowledge_base_id,
                        "modelArn": self.model_arn,
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": DEFAULT_TOP_K,
                                "overrideSearchType": "HYBRID",
                            }
                        },
                        "generationConfiguration": {
                            "promptTemplate": {
                                "textPromptTemplate": prompt_template,
                            },
                            "inferenceConfig": {
                                "textInferenceConfig": {
                                    "temperature": GENERATION_TEMPERATURE,
                                    "topP": GENERATION_TOP_P,
                                    "maxTokens": GENERATION_MAX_TOKENS,
                                }
                            },
                        },
                    },
                },
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            logger.exception("retrieve_and_generate() failed")
            raise RuntimeError(f"Retrieve & generate failed ({code})") from e
 
    def retrieve_and_generate_with_citations(
        self,
        question: str,
        prompt_template: str = GENERATION_PROMPT_TEMPLATE,
    ) -> dict:
        response = self.retrieve_and_generate(question, prompt_template)
        return {
            "answer": response.get("output", {}).get("text", ""),
            "citations": response.get("citations", []),
        }
 
    def stream_retrieve_and_generate(
        self,
        question: str,
        prompt_template: str = GENERATION_PROMPT_TEMPLATE,
    ):
        """Reasoning path only. Never use this for the verbatim path."""
        try:
            response = self.client.retrieve_and_generate_stream(
                input={"text": question},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.knowledge_base_id,
                        "modelArn": self.model_arn,
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": {
                                "numberOfResults": DEFAULT_TOP_K,
                                "overrideSearchType": "HYBRID",
                            }
                        },
                        "generationConfiguration": {
                            "promptTemplate": {
                                "textPromptTemplate": prompt_template,
                            },
                            "inferenceConfig": {
                                "textInferenceConfig": {
                                    "temperature": GENERATION_TEMPERATURE,
                                    "topP": GENERATION_TOP_P,
                                    "maxTokens": GENERATION_MAX_TOKENS,
                                }
                            },
                        },
                    },
                },
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            logger.exception("retrieve_and_generate_stream() failed")
            raise RuntimeError(f"Streaming failed ({code})") from e
 
        for event in response["stream"]:
            if "output" in event:
                text = event["output"].get("text", "")
                if text:
                    yield text
 
 
# Singleton instance
_kb = BedrockKnowledgeBase()
 
 
def knowledge_base_search(question: str, top_k: int = DEFAULT_TOP_K) -> list:
    return _kb.retrieve_only(question, top_k=top_k)
 
 
def knowledge_base_answer(question: str) -> dict:
    return _kb.retrieve_and_generate(question)
 
 
def knowledge_base_answer_with_citations(question: str) -> dict:
    return _kb.retrieve_and_generate_with_citations(question)
