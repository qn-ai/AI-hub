# ---- Planning Letter revision (direct model call, no retrieval) --------
 
    def revise_text(
        self,
        draft: str,
        system_prompt: str = PLANNING_LETTER_SYSTEM_PROMPT,
    ) -> str:
        """Revise pasted draft text with Nova Pro. Does NOT touch the KB."""
        try:
            response = self.runtime_client.converse(
                modelId=self.model_arn,
                system=[{"text": system_prompt}],
                messages=[
                    {"role": "user", "content": [{"text": draft}]},
                ],
                inferenceConfig={
                    "temperature": REVISION_TEMPERATURE,
                    "topP": REVISION_TOP_P,
                    "maxTokens": REVISION_MAX_TOKENS,
                },
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            logger.exception("converse() failed")
            raise RuntimeError(f"Revision failed ({code})") from e
 
        content = (
            response.get("output", {})
            .get("message", {})
            .get("content", [])
        )
        return content[0].get("text", "") if content else ""
