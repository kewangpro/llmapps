"""
Orchestrator Agent - Main orchestrator agent that coordinates sub-agents
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from .base_agent import BaseAgent
from .file_search_agent import FileSearchAgent
from .web_search_agent import WebSearchAgent
from .system_info_agent import SystemInfoAgent
from .code_analysis_agent import CodeAnalysisAgent
from .data_processing_agent import DataProcessingAgent
from .presentation_agent import PresentationAgent
from .image_analysis_agent import ImageAnalysisAgent
from .stock_analysis_agent import StockAnalysisAgent

logger = logging.getLogger("MultiAgentSystem")


class OrchestratorAgent(BaseAgent):
    """Main orchestrator agent that coordinates sub-agents"""

    def __init__(self, model: str = "gemma3:latest"):
        super().__init__(model)
        self.sub_agents = {
            "file_search": FileSearchAgent(model),
            "web_search": WebSearchAgent(model),
            "system_info": SystemInfoAgent(model),
            "code_analysis": CodeAnalysisAgent(model),
            "data_processing": DataProcessingAgent(model),
            "presentation": PresentationAgent(model),
            "image_analysis": ImageAnalysisAgent(model),
            "stock_analysis": StockAnalysisAgent(model)
        }

    def _select_agents(self, query: str, available_tools: List[str], attached_file: Dict = None) -> List[str]:
        """Select which sub-agents to use based on query and available tools"""
        try:
            logger.info(f"\n🧠 ORCHESTRATOR ANALYZING QUERY: '{query}'")
            logger.info(f"📋 Available tools: {available_tools}")

            tools_desc = {
                "file_search": "search for files and directories",
                "web_search": "search the internet for information",
                "system_info": "get system information (CPU, memory, disk, network)",
                "code_analysis": "analyze code for quality, security, performance",
                "data_processing": "process, analyze, or transform data",
                "presentation": "generate PowerPoint presentations from text or files",
                "image_analysis": "analyze image files for content, text, and metadata",
                "stock_analysis": "analyze stock market data and performance using Yahoo Finance"
            }

            available_desc = [f"- {tool}: {tools_desc[tool]}" for tool in available_tools if tool in tools_desc]

            # Add file context if available
            file_context = ""
            if attached_file:
                file_type = attached_file.get("type", "")
                file_name = attached_file.get("name", "")
                if file_type.startswith("image/") or any(file_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg']):
                    file_context = f"\nIMPORTANT: User has uploaded an IMAGE file ({file_name}). For image analysis queries, prioritize 'image_analysis' tool."
                elif file_type.startswith("text/") or any(file_name.lower().endswith(ext) for ext in ['.txt', '.md', '.csv', '.json', '.xml']):
                    file_context = f"\nIMPORTANT: User has uploaded a TEXT/DATA file ({file_name}). For data processing queries, prioritize 'data_processing' tool."

            prompt = f"""Given this user query: "{query}"{file_context}

Available tools:
{chr(10).join(available_desc)}

Which tools should be used? Consider:
1. What information is the user asking for?
2. What actions need to be performed?
3. Are multiple tools needed?
4. If an image file is attached, prioritize image_analysis for visual analysis
5. If a data file is attached, prioritize data_processing for data analysis

Important: If this is a conversational query (greetings, thanks, general chat) that doesn't require any tools, respond with "NONE".

Respond with:
- Tool names (one per line) if tools are needed
- "NONE" if no tools are required for this conversational query"""

            logger.info("🤔 Orchestrator thinking...")
            response = self.llm.call(prompt)
            logger.info(f"💭 Orchestrator reasoning: {response.strip()}")

            # Parse response and validate against available tools
            selected = []
            for line in response.strip().split('\n'):
                tool = line.strip().lower().replace('-', '').replace('*', '').strip()
                if tool in available_tools:
                    selected.append(tool)

            # Check if any valid tools were selected
            if selected:
                logger.info(f"✅ Selected agents: {selected}")
            elif "NONE" in response.strip().upper():
                selected = []
                logger.info("✅ No tools needed for conversational query")
            else:
                # Fallback: if no valid selection and not explicitly NONE, use first available tool
                if available_tools:
                    selected = [available_tools[0]]
                    logger.info(f"⚠️  Fallback selection: {selected}")
                else:
                    logger.info("✅ No tools available")

            return selected

        except Exception as e:
            logger.error(f"❌ Agent selection error: {str(e)}")
            # Fallback: use first available tool
            fallback = [available_tools[0]] if available_tools else []
            logger.info(f"🔄 Emergency fallback: {fallback}")
            return fallback

    def _generate_conversational_response(self, query: str) -> str:
        """Generate a conversational response for queries that don't need tools"""
        try:
            prompt = f"""Respond to this conversational query: "{query}"

This is a simple greeting or conversational message that doesn't require any tools or data retrieval.
Provide a friendly, helpful response as an AI assistant.

Keep the response concise and natural."""

            response = self.llm.call(prompt).strip()
            logger.info(f"💬 Generated conversational response: '{response}'")
            return response
        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            # Fallback responses
            greetings = ["hello", "hi", "hey", "greetings"]
            thanks = ["thank", "thanks", "thx"]

            query_lower = query.lower()
            if any(greeting in query_lower for greeting in greetings):
                fallback = "Hello! I'm here to help you with various tasks using my available tools. What can I assist you with today?"
            elif any(thank in query_lower for thank in thanks):
                fallback = "You're welcome! Feel free to ask if you need help with anything else."
            else:
                fallback = "I'm here to help! What would you like me to do?"

            logger.info(f"💬 Using fallback conversational response: '{fallback}'")
            return fallback

    def _generate_initial_response(self, query: str, selected_agents: List[str], attached_file: Dict = None) -> str:
        """Generate initial acknowledgment response from orchestrator"""
        try:
            # Build context about what we're going to do
            context = f"Query: '{query}'\n"
            context += f"Selected agents: {selected_agents}\n"

            if attached_file:
                file_name = attached_file.get("filename", "unknown file")
                context += f"Attached file: {file_name}\n"

            prompt = f"""You are the OrchestratorAgent. Provide a brief, professional acknowledgment of what you're about to do.

{context}

Generate a short response that:
1. Acknowledges the user's request
2. Mentions the attached file if present
3. Briefly explains what tool(s) you'll use
4. Keeps it concise (1-2 sentences)

Examples:
- "I'll analyze the attached image using my image analysis capabilities."
- "Let me search for those files and analyze the code for you."
- "I'll process the attached data file and provide you with insights."

Respond as the orchestrator agent in first person."""

            return self.llm.call(prompt).strip()
        except Exception as e:
            logger.error(f"Error generating initial response: {e}")
            # Fallback response
            if attached_file:
                file_name = attached_file.get("filename", "file")
                return f"I'll analyze the attached {file_name} using the appropriate tools."
            else:
                return f"I'll help you with that using my available tools."

    async def execute_with_callback(self, query: str, available_tools: List[str] = None, attached_file: Dict = None, callback=None) -> Dict[str, Any]:
        """Execute with real-time callback for immediate messaging"""
        try:
            logger.info(f"\n🎯 ORCHESTRATOR STARTING EXECUTION WITH CALLBACK")
            logger.info(f"Query: '{query}'")

            if not available_tools:
                available_tools = list(self.sub_agents.keys())

            # Select appropriate sub-agents
            selected_agents = self._select_agents(query, available_tools, attached_file)

            results = []
            final_answer = ""
            initial_response = ""

            # Handle conversational queries that don't need tools
            if not selected_agents:
                logger.info("💬 Conversational query - no tools needed")
                final_answer = self._generate_conversational_response(query)
                logger.info(f"💬 Sending conversational response via callback")
                if callback:
                    await callback("initial_response", final_answer)
                return {
                    "orchestrator": "OrchestratorAgent",
                    "query": query,
                    "selected_agents": [],
                    "agent_results": [],
                    "initial_response": final_answer,
                    "final_answer": final_answer,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }

            # Generate and send initial response immediately
            logger.info("🎤 Generating initial orchestrator response...")
            initial_response = self._generate_initial_response(query, selected_agents, attached_file)
            logger.info(f"📢 Initial response: {initial_response}")

            if callback:
                logger.info("📤 Sending initial response via callback")
                await callback("initial_response", initial_response)

                # CRITICAL: Add a small pause to allow frontend to render the initial response
                # before continuing with tool execution
                import asyncio
                await asyncio.sleep(0.1)  # 100ms pause for frontend to render

            logger.info(f"\n🚀 EXECUTING {len(selected_agents)} SUB-AGENTS...")

            # Execute selected sub-agents sequentially with context
            for i, agent_name in enumerate(selected_agents, 1):
                if agent_name in self.sub_agents:
                    logger.info(f"\n🤖 SUB-AGENT {i}/{len(selected_agents)}: {agent_name.upper()}Agent")

                    # Build context-aware query for agents after the first one
                    if i == 1:
                        # First agent uses original query
                        agent_query = query
                        # Add attached file information for relevant agents
                        if attached_file and agent_name in ["image_analysis", "data_processing", "presentation"]:
                            file_path = attached_file.get("path", "")
                            if file_path:
                                agent_query += f" FILE_PATH:{file_path}"
                                logger.info(f"📎 Added file attachment: {file_path}")
                        logger.info(f"🎯 Delegating original query to {agent_name} specialist...")
                    else:
                        # Subsequent agents get context from previous results
                        logger.info(f"🔗 Building context-aware query for {agent_name} based on previous results...")

                        context = "Previous agent results:\n"
                        for j, prev_result in enumerate(results, 1):
                            if prev_result.get("success"):
                                context += f"{j}. {prev_result.get('agent', 'Unknown')}: {json.dumps(prev_result.get('result', {}), indent=2)}\n\n"

                        # Let the LLM create a contextual query for this agent
                        context_prompt = f"""Original user query: "{query}"

{context}

Create a specific query for the {agent_name} agent that uses the ACTUAL data from previous results.

IMPORTANT: Extract specific values from the previous results and use them directly in your query.

Examples:
- If system_info shows "macOS Darwin 25.0.0", search for "latest macOS Darwin 25.0.0 updates"
- If system_info shows "Ubuntu 22.04", search for "Ubuntu 22.04 latest version updates"
- If file_search found "auth.py, login.py", analyze those specific files

For {agent_name} agent, create a query that uses the specific data values from the previous results above.
Do NOT use placeholders like [OS version] - use the actual values.

Respond with just the refined query, no additional text."""

                        agent_query = self.llm.call(context_prompt).strip()
                        logger.info(f"🎯 Context-aware query for {agent_name}: '{agent_query}'")

                    agent_result = self.sub_agents[agent_name].execute(agent_query)

                    if agent_result.get("success"):
                        logger.info(f"✅ {agent_name.upper()}Agent completed successfully")
                        logger.info(f"🔧 Tool used: {agent_result.get('tool', 'unknown')}")
                        logger.info(f"📊 Result preview: {str(agent_result.get('result', {}))[:100]}...")
                    else:
                        logger.error(f"❌ {agent_name.upper()}Agent failed: {agent_result.get('error', 'Unknown error')}")

                    results.append(agent_result)

            # Generate final answer based on all results
            if results:
                logger.info(f"\n🧠 SYNTHESIZING FINAL ANSWER...")
                logger.info(f"📝 Combining results from {len(results)} agents...")

                # Check if any agent has pre-formatted insights (like stock analysis)
                formatted_insights = []
                raw_results = []

                for result in results:
                    if result.get("result", {}).get("ai_insights", {}).get("analysis"):
                        # Use the beautifully formatted analysis directly (for stock analysis, etc.)
                        insight = result["result"]["ai_insights"]["analysis"]
                        formatted_insights.append(insight)
                        logger.info(f"📊 Using pre-formatted insight from {result.get('agent', 'unknown')} agent")
                    elif result.get("agent") == "ImageAnalysisAgent" and result.get("result", {}).get("analysis"):
                        # Use image analysis directly without further synthesis
                        insight = result["result"]["analysis"]
                        formatted_insights.append(insight)
                        logger.info(f"🖼️ Using image analysis directly from {result.get('agent', 'unknown')} agent")
                    else:
                        # Keep raw results for other agents
                        raw_results.append(result)

                if formatted_insights and not raw_results:
                    # If we only have formatted insights, use them directly
                    final_answer = "\n\n".join(formatted_insights)
                    logger.info(f"✅ Using pre-formatted insights directly ({len(final_answer)} characters)")
                elif formatted_insights and raw_results:
                    # Mix formatted insights with synthesis of raw results
                    prompt = f"""The user asked: "{query}"

I have some pre-formatted analysis and some raw tool results to combine:

PRE-FORMATTED ANALYSIS:
{chr(10).join(formatted_insights)}

RAW TOOL RESULTS:
{json.dumps(raw_results, indent=2)}

Please provide a brief introduction and then present the pre-formatted analysis, followed by any additional insights from the raw results. Keep the pre-formatted sections intact."""

                    final_answer = self.llm.call(prompt)
                    logger.info(f"✅ Combined formatted insights with synthesis ({len(final_answer)} characters)")
                else:
                    # Traditional synthesis for agents without formatted insights
                    prompt = f"""Based on these tool execution results, provide a comprehensive answer to: "{query}"

Results:
{json.dumps(results, indent=2)}

Provide a clear, helpful response that synthesizes the information from the tools."""

                    final_answer = self.llm.call(prompt)
                    logger.info(f"✅ Final answer generated via synthesis ({len(final_answer)} characters)")

                # Send final answer via callback
                if callback:
                    logger.info("📤 Sending final answer via callback")
                    await callback("final_answer", final_answer)

            logger.info(f"\n🎉 ORCHESTRATION COMPLETE!")
            logger.info(f"📊 Total agents used: {len(selected_agents)}")
            logger.info(f"🎯 Successful executions: {sum(1 for r in results if r.get('success'))}")

            return {
                "orchestrator": "OrchestratorAgent",
                "query": query,
                "selected_agents": selected_agents,
                "agent_results": results,
                "initial_response": initial_response,
                "final_answer": final_answer,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ ORCHESTRATION FAILED: {str(e)}")
            return {
                "orchestrator": "OrchestratorAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def execute(self, query: str, available_tools: List[str] = None, attached_file: Dict = None) -> Dict[str, Any]:
        """Orchestrate sub-agents to handle the user query"""
        try:
            logger.info(f"\n🎯 ORCHESTRATOR STARTING EXECUTION")
            logger.info(f"Query: '{query}'")

            if not available_tools:
                available_tools = list(self.sub_agents.keys())

            # Select appropriate sub-agents
            selected_agents = self._select_agents(query, available_tools, attached_file)

            results = []
            final_answer = ""
            initial_response = ""

            # Handle conversational queries that don't need tools
            if not selected_agents:
                logger.info("💬 Conversational query - no tools needed")
                final_answer = self._generate_conversational_response(query)
                logger.info(f"💬 Returning conversational result with final_answer: '{final_answer}'")
                return {
                    "orchestrator": "OrchestratorAgent",
                    "query": query,
                    "selected_agents": [],
                    "agent_results": [],
                    "initial_response": final_answer,  # Same for conversational
                    "final_answer": final_answer,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }

            # Generate initial response from orchestrator
            logger.info("🎤 Generating initial orchestrator response...")
            initial_response = self._generate_initial_response(query, selected_agents, attached_file)
            logger.info(f"📢 Initial response: {initial_response}")

            logger.info(f"\n🚀 EXECUTING {len(selected_agents)} SUB-AGENTS...")

            # Execute selected sub-agents sequentially with context
            for i, agent_name in enumerate(selected_agents, 1):
                if agent_name in self.sub_agents:
                    logger.info(f"\n🤖 SUB-AGENT {i}/{len(selected_agents)}: {agent_name.upper()}Agent")

                    # Build context-aware query for agents after the first one
                    if i == 1:
                        # First agent uses original query
                        agent_query = query
                        # Add attached file information for relevant agents
                        if attached_file and agent_name in ["image_analysis", "data_processing", "presentation"]:
                            file_path = attached_file.get("path", "")
                            if file_path:
                                agent_query += f" FILE_PATH:{file_path}"
                                logger.info(f"📎 Added file attachment: {file_path}")
                        logger.info(f"🎯 Delegating original query to {agent_name} specialist...")
                    else:
                        # Subsequent agents get context from previous results
                        logger.info(f"🔗 Building context-aware query for {agent_name} based on previous results...")

                        context = "Previous agent results:\n"
                        for j, prev_result in enumerate(results, 1):
                            if prev_result.get("success"):
                                context += f"{j}. {prev_result.get('agent', 'Unknown')}: {json.dumps(prev_result.get('result', {}), indent=2)}\n\n"

                        # Let the LLM create a contextual query for this agent
                        context_prompt = f"""Original user query: "{query}"

{context}

Create a specific query for the {agent_name} agent that uses the ACTUAL data from previous results.

IMPORTANT: Extract specific values from the previous results and use them directly in your query.

Examples:
- If system_info shows "macOS Darwin 25.0.0", search for "latest macOS Darwin 25.0.0 updates"
- If system_info shows "Ubuntu 22.04", search for "Ubuntu 22.04 latest version updates"
- If file_search found "auth.py, login.py", analyze those specific files

For {agent_name} agent, create a query that uses the specific data values from the previous results above.
Do NOT use placeholders like [OS version] - use the actual values.

Respond with just the refined query, no additional text."""

                        agent_query = self.llm.call(context_prompt).strip()
                        logger.info(f"🎯 Context-aware query for {agent_name}: '{agent_query}'")

                    agent_result = self.sub_agents[agent_name].execute(agent_query)

                    if agent_result.get("success"):
                        logger.info(f"✅ {agent_name.upper()}Agent completed successfully")
                        logger.info(f"🔧 Tool used: {agent_result.get('tool', 'unknown')}")
                        logger.info(f"📊 Result preview: {str(agent_result.get('result', {}))[:100]}...")
                    else:
                        logger.error(f"❌ {agent_name.upper()}Agent failed: {agent_result.get('error', 'Unknown error')}")

                    results.append(agent_result)

            # Generate final answer based on all results
            if results:
                logger.info(f"\n🧠 SYNTHESIZING FINAL ANSWER...")
                logger.info(f"📝 Combining results from {len(results)} agents...")

                # Check if any agent has pre-formatted insights (like stock analysis)
                formatted_insights = []
                raw_results = []

                for result in results:
                    if result.get("result", {}).get("ai_insights", {}).get("analysis"):
                        # Use the beautifully formatted analysis directly (for stock analysis, etc.)
                        insight = result["result"]["ai_insights"]["analysis"]
                        formatted_insights.append(insight)
                        logger.info(f"📊 Using pre-formatted insight from {result.get('agent', 'unknown')} agent")
                    elif result.get("agent") == "ImageAnalysisAgent" and result.get("result", {}).get("analysis"):
                        # Use image analysis directly without further synthesis
                        insight = result["result"]["analysis"]
                        formatted_insights.append(insight)
                        logger.info(f"🖼️ Using image analysis directly from {result.get('agent', 'unknown')} agent")
                    else:
                        # Keep raw results for other agents
                        raw_results.append(result)

                if formatted_insights and not raw_results:
                    # If we only have formatted insights, use them directly
                    final_answer = "\n\n".join(formatted_insights)
                    logger.info(f"✅ Using pre-formatted insights directly ({len(final_answer)} characters)")
                elif formatted_insights and raw_results:
                    # Mix formatted insights with synthesis of raw results
                    prompt = f"""The user asked: "{query}"

I have some pre-formatted analysis and some raw tool results to combine:

PRE-FORMATTED ANALYSIS:
{chr(10).join(formatted_insights)}

RAW TOOL RESULTS:
{json.dumps(raw_results, indent=2)}

Please provide a brief introduction and then present the pre-formatted analysis, followed by any additional insights from the raw results. Keep the pre-formatted sections intact."""

                    final_answer = self.llm.call(prompt)
                    logger.info(f"✅ Combined formatted insights with synthesis ({len(final_answer)} characters)")
                else:
                    # Traditional synthesis for agents without formatted insights
                    prompt = f"""Based on these tool execution results, provide a comprehensive answer to: "{query}"

Results:
{json.dumps(results, indent=2)}

Provide a clear, helpful response that synthesizes the information from the tools."""

                    final_answer = self.llm.call(prompt)
                    logger.info(f"✅ Final answer generated via synthesis ({len(final_answer)} characters)")

            logger.info(f"\n🎉 ORCHESTRATION COMPLETE!")
            logger.info(f"📊 Total agents used: {len(selected_agents)}")
            logger.info(f"🎯 Successful executions: {sum(1 for r in results if r.get('success'))}")

            return {
                "orchestrator": "OrchestratorAgent",
                "query": query,
                "selected_agents": selected_agents,
                "agent_results": results,
                "initial_response": initial_response,
                "final_answer": final_answer,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ ORCHESTRATION FAILED: {str(e)}")
            return {
                "orchestrator": "OrchestratorAgent",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }