# core/llm_client.py

import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMClient:
    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.model = os.getenv("MODEL_DEPLOYMENT_NAME")
        
        # Validate required environment variables
        if not self.azure_endpoint:
            raise ValueError("AZURE_ENDPOINT not found in .env file")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in .env file")
        if not self.api_version:
            raise ValueError("AZURE_OPENAI_API_VERSION not found in .env file")
        if not self.model:
            raise ValueError("MODEL_DEPLOYMENT_NAME not found in .env file")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
    
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0) -> str:
        """
        Centralized LLM call with error handling
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            
            # Optional: Log token usage
            if response.usage:
                print(f"Prompt Tokens: {response.usage.prompt_tokens}")
                print(f"Completion Tokens: {response.usage.completion_tokens}")
                print(f"Total Tokens: {response.usage.total_tokens}")
            
            return response.choices[0].message.content
        
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}")