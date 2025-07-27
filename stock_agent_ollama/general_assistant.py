"""
General Assistant Tool - LLM-powered fallback for non-stock questions
Provides general information and assistance using Ollama LLM
"""

import logging
from typing import Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class GeneralAssistantInput(BaseModel):
    question: str = Field(description="General question or request from the user")


class GeneralAssistant(BaseTool):
    name = "general_assistant"
    description = """Use this tool for general questions, conversations, or requests that are not related to stock analysis.
    This tool can answer questions about finance concepts, market terminology, investment strategies, or any general topics.
    Input should be JSON format: {"question": "What is the difference between stocks and bonds?"}"""
    args_schema = GeneralAssistantInput
    
    def __init__(self):
        super().__init__()
        # Initialize Ollama LLM using the same model as the main agent
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM to avoid Pydantic field conflicts"""
        try:
            llm = Ollama(
                model="gemma3:latest",  # Same model as main agent
                temperature=0.7,  # Slightly higher for more creative responses
                verbose=False,
                timeout=60
            )
            # Store in __dict__ to avoid Pydantic validation
            self.__dict__['_llm'] = llm
            logger.info("✅ General Assistant LLM initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            self.__dict__['_llm'] = None
    
    def _get_llm(self):
        """Get the LLM instance"""
        return self.__dict__.get('_llm', None)
    
    def _run(self, question: str) -> Dict[str, Any]:
        """
        Handle general questions using LLM
        """
        try:
            logger.info(f"General Assistant processing question: {question[:100]}{'...' if len(question) > 100 else ''}")
            
            llm = self._get_llm()
            if not llm:
                return {
                    "success": False,
                    "error": "LLM not available",
                    "response": "I'm sorry, but I'm currently unable to process general questions. Please try asking about stock analysis instead."
                }
            
            # Create a helpful prompt for general assistance
            prompt = PromptTemplate.from_template("""
You are a helpful financial and general knowledge assistant. You can answer questions about:
- Financial concepts and terminology
- Investment strategies and market basics
- General knowledge topics
- Technology and business concepts

Please provide a clear, helpful, and accurate response to the following question:

Question: {question}

Response:""")
            
            # Generate response using LLM
            formatted_prompt = prompt.format(question=question)
            response = llm.invoke(formatted_prompt)
            
            logger.info("✅ General Assistant response generated successfully")
            
            return {
                "success": True,
                "question": question,
                "response": response.strip(),
                "tool_used": "general_assistant",
                "complete": True  # Signal that this is a complete response
            }
            
        except Exception as e:
            logger.error(f"❌ Error in General Assistant: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or ask about stock analysis instead."
            }
    
    async def _arun(self, question: str) -> Dict[str, Any]:
        """Async version - not implemented for now"""
        return self._run(question)


# Create tool instance
def get_general_assistant_tool():
    """Factory function to create GeneralAssistant tool instance"""
    return GeneralAssistant()


if __name__ == "__main__":
    # Test the tool
    tool = get_general_assistant_tool()
    test_questions = [
        "What is the difference between stocks and bonds?",
        "How does compound interest work?",
        "What is machine learning?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = tool._run(question)
        print(f"Response: {result.get('response', 'No response')}")
        print("-" * 50)