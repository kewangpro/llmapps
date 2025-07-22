from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from tools import search_tool
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchAgent:
    """Interactive search agent with conversation memory"""
    
    def __init__(self, llm: BaseChatModel, msgs: StreamlitChatMessageHistory, memory: ConversationBufferMemory):
        self.llm = llm
        self.msgs = msgs
        self.memory = memory
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the conversational chat agent"""
        tools = [search_tool]
        self.chat_agent = ConversationalChatAgent.from_llm_and_tools(
            llm=self.llm, 
            tools=tools
        )
        
        self.executor = AgentExecutor.from_agent_and_tools(
            agent=self.chat_agent,
            tools=tools,
            memory=self.memory,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )
    
    def process(self, message: str) -> str:
        """Process a message and return the response"""
        try:
            response = self.executor.invoke({"input": message})
            return response.get("output", "No response generated")
        except Exception as e:
            return f"Error processing search request: {str(e)}"
    
    def get_intermediate_steps(self, message: str):
        """Get intermediate steps for detailed view"""
        try:
            response = self.executor.invoke({"input": message})
            return response.get("intermediate_steps", [])
        except Exception as e:
            return []