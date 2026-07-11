from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
import os
import langchain

# Removed custom logging - using LangChain debug only

# Import components from different agent modes
from plan_execute import PlanExecuteAgent
from supervisor_agents import SupervisorAgent
from search_agent import SearchAgent
from hybrid_agent import HybridAgent

class AgentManager:
    """Central manager for coordinating different agent modes"""
    
    def __init__(self, llm_provider="openai", debug_mode=True):
        # Enable LangChain debug mode to show agent thoughts
        if debug_mode:
            langchain.debug = True
        # Initialize LLM based on provider selection
        if llm_provider == "openai":
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                self.llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)
                self.current_llm = "OpenAI GPT-4o-mini"
            else:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        elif llm_provider == "ollama":
            try:
                # Try models in order of preference (tool support priority)
                models_to_try = [
                    ("llama3.2:latest", "Ollama Llama3.2"),
                    ("mistral:latest", "Ollama Mistral"), 
                    ("gemma3:12b", "Ollama Gemma3:12b"),
                    ("gemma3:latest", "Ollama Gemma3")
                ]
                
                for model_name, display_name in models_to_try:
                    try:
                        # Test if model exists and supports basic chat
                        test_llm = ChatOllama(
                            model=model_name,
                            base_url="http://localhost:11434"
                        )
                        # Simple test to verify model works
                        test_response = test_llm.invoke("Hello")
                        
                        self.llm = test_llm
                        self.current_llm = display_name
                        break
                    except Exception as model_error:
                        continue
                else:
                    raise ValueError("No working Ollama models found")
                    
            except Exception as e:
                raise ValueError(f"Failed to connect to Ollama. Please ensure Ollama is running. Error: {e}")
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        # LLM initialized successfully
        
        # Initialize message history
        self.msgs = StreamlitChatMessageHistory()
        self.memory = ConversationBufferMemory(
            chat_memory=self.msgs,
            return_messages=True,
            memory_key="chat_history",
            output_key="output"
        )
        
        # Initialize different agent modes
        self.agents = {
            "Plan & Execute": PlanExecuteAgent(self.llm),
            "Multi-Agent Supervisor": SupervisorAgent(self.llm),
            "Interactive Search": SearchAgent(self.llm, self.msgs, self.memory),
            "Hybrid Mode": HybridAgent(self.llm)
        }
    
    def get_current_llm(self) -> str:
        """Get current LLM provider info"""
        return self.current_llm
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get formatted message history"""
        messages = []
        for msg in self.msgs.messages:
            messages.append({
                "role": "ai" if msg.type == "ai" else "user",
                "content": msg.content
            })
        return messages
    
    def clear_messages(self):
        """Clear message history"""
        self.msgs.clear()
        self.msgs.add_ai_message("How can I help you?")
    
    def get_graph(self, mode: str):
        """Get the agent graph for visualization"""
        if mode in self.agents and hasattr(self.agents[mode], 'get_graph'):
            return self.agents[mode].get_graph()
        return None
    
    def process_message(self, message: str, mode: str, response_container=None) -> str:
        """Process a message with the selected agent mode"""
        self.msgs.add_user_message(message)
        
        try:
            agent = self.agents[mode]
            
            if mode == "Interactive Search":
                # Special handling for search agent with memory
                response = agent.process(message)
            elif response_container is not None:
                # For graph-based agents, show streaming details
                response = agent.process_with_streaming(message, response_container)
            else:
                # Standard processing
                response = agent.process(message)
            
            # Add AI response to memory
            self.msgs.add_ai_message(response)
            return response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            self.msgs.add_ai_message(error_msg)
            return error_msg