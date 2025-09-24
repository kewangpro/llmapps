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
from .cost_analysis_agent import CostAnalysisAgent
from .data_processing_agent import DataProcessingAgent
from .presentation_agent import PresentationAgent
from .image_analysis_agent import ImageAnalysisAgent
from .stock_analysis_agent import StockAnalysisAgent
from .visualization_agent import VisualizationAgent
from .forecast_agent import ForecastAgent

logger = logging.getLogger("OrchestratorAgent")


class OrchestratorAgent(BaseAgent):
    """Main orchestrator agent that coordinates sub-agents"""

    def __init__(self):
        super().__init__()
        self.sub_agents = {
            "file_search": FileSearchAgent(),
            "web_search": WebSearchAgent(),
            "system_info": SystemInfoAgent(),
            "cost_analysis": CostAnalysisAgent(),
            "data_processing": DataProcessingAgent(),
            "presentation": PresentationAgent(),
            "image_analysis": ImageAnalysisAgent(),
            "stock_analysis": StockAnalysisAgent(),
            "visualization": VisualizationAgent(),
            "forecast": ForecastAgent()
        }

    def _select_agents(self, query: str, available_tools: List[str], attached_file: Dict = None) -> List[str]:
        """Select which sub-agents to use based on query and available tools"""
        try:
            logger.info(f"\n🧠 ORCHESTRATOR ANALYZING QUERY: '{query}'")
            logger.info(f"📋 Available tools: {available_tools}")

            # Get tool descriptions from MultiAgentSystem
            from multi_agent_system import MultiAgentSystem
            all_tools = MultiAgentSystem.get_available_tools()
            tools_desc = {tool["name"]: tool["short_description"] for tool in all_tools}

            available_desc = [f"- {tool}: {tools_desc[tool]}" for tool in available_tools if tool in tools_desc]

            # Add file context if available
            file_context = ""
            if attached_file:
                file_type = attached_file.get("type", "")
                file_name = attached_file.get("name", "")
                if file_type.startswith("image/") or any(file_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg']):
                    file_context = f"\nIMPORTANT: User has uploaded an IMAGE file ({file_name}). For image analysis queries, prioritize 'image_analysis' tool."
                elif file_type.startswith("text/") or any(file_name.lower().endswith(ext) for ext in ['.txt', '.md', '.csv', '.json', '.xml']):
                    file_context = f"\nIMPORTANT: User has uploaded a TEXT/DATA file ({file_name}). For data processing queries, prioritize 'data_processing' tool. For chart/visualization requests, prioritize 'visualization' tool."

            prompt = f"""Given this user query: "{query}"{file_context}

Available tools:
{chr(10).join(available_desc)}

Which tools should be used? Consider:
1. What information is the user asking for?
2. What actions need to be performed?
3. Are multiple tools needed?
4. If an image file is attached, prioritize image_analysis for visual analysis
5. If a data file is attached, prioritize data_processing for data analysis

TOOL SELECTION PRIORITY RULES:
1. STOCK QUERIES (company names like MSFT/AAPL, stock symbols, market analysis, stock performance):
   - Use "stock_analysis" AND "visualization" tools together
   - Examples: "show AAPL performance", "analyze Tesla stock", "get financial metrics for MSFT"

2. COST/SPENDING QUERIES (COGS analysis, business unit costs, spending patterns):
   - Use "cost_analysis" AND "visualization" tools together
   - Examples: "show cost trends", "analyze spending per business unit", "cost per department"

3. PRIORITIZATION: If query contains stock symbols (MSFT, AAPL, etc.) or company names, treat as STOCK QUERY first
4. AVOID DUPLICATES: Select each tool only once, even if multiple categories match

EXECUTION ORDER RULES:
1. For forecast + visualization: Run forecast BEFORE visualization so charts show both historical and predicted data
2. For stock analysis + forecast + visualization: Run stock_analysis → forecast → visualization
3. Data-producing tools (stock_analysis, forecast, cost_analysis) should run before visualization

Important: If this is a conversational query (greetings, thanks, general chat) that doesn't require any tools, respond with "NONE".

Respond with:
- Tool names (one per line) IN EXECUTION ORDER if tools are needed
- "NONE" if no tools are required for this conversational query"""

            logger.info("🤔 Orchestrator thinking...")
            response = self.llm.call(prompt)
            logger.info(f"💭 Orchestrator reasoning: {response.strip()}")

            # Parse response and validate against available tools (avoid duplicates)
            selected = []
            for line in response.strip().split('\n'):
                tool = line.strip().lower().replace('-', '').replace('*', '').strip()
                if tool in available_tools and tool not in selected:
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
                        if attached_file and agent_name in ["image_analysis", "data_processing", "presentation", "visualization", "cost_analysis"]:
                            file_path = attached_file.get("path", "")
                            logger.info(f"🔍 Debug attached_file: {attached_file}")
                            logger.info(f"🔍 Debug file_path: '{file_path}'")
                            if file_path:
                                agent_query += f" FILE_PATH:{file_path}"
                                logger.info(f"📎 Added file attachment: {file_path}")
                            else:
                                logger.warning(f"⚠️ No file path found in attached_file for {agent_name}")
                        logger.info(f"🎯 Delegating original query to {agent_name} specialist...")
                    else:
                        # Subsequent agents get context from previous results
                        logger.info(f"🔗 Building context-aware query for {agent_name} based on previous results...")

                        # Handle forecast tool when following any data-producing agent
                        if agent_name == "forecast":
                            # Find any previous agent that produced tool_data (generic detection)
                            data_agent = self._find_previous_data_agent(results)

                            if data_agent:
                                # Export tool_data to file (pure file I/O, no data processing)
                                file_path = self._export_tool_data_to_file(data_agent["result"]["tool_data"])

                                if file_path:
                                    agent_query = f"predict future trends using LSTM neural network FILE_PATH:{file_path}"
                                    logger.info(f"🎯 Exported data to file {file_path} for forecast")
                                else:
                                    agent_query = f"No data available for forecasting from previous agent"
                            else:
                                agent_query = f"predict future trends based on: {query}"
                        # Handle visualization tool when following any data-producing agent
                        elif agent_name == "visualization":
                            # Find any previous agent that produced tool_data (generic detection)
                            data_agent = self._find_previous_data_agent(results)

                            if data_agent:
                                # Export tool_data to file (pure file I/O, no data processing)
                                file_path = self._export_tool_data_to_file(data_agent["result"]["tool_data"])

                                if file_path:
                                    # Generate generic visualization query based on user's ask
                                    viz_query = self._generate_generic_visualization_query(query)
                                    agent_query = f"{viz_query} FILE_PATH:{file_path}"
                                    logger.info(f"🎯 Exported data to file {file_path} for visualization")
                                else:
                                    agent_query = f"No data available for visualization from previous agent"
                            else:
                                agent_query = f"Create a visualization for: {query}"
                        else:
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

                # Separate analysis results from visualization results
                analysis_content = []
                non_analysis_results = []

                for result in results:
                    if result.get("agent") == "VisualizationAgent":
                        # Visualization results are displayed separately, not included in final answer
                        logger.info(f"📊 Visualization result will be displayed separately")
                        continue
                    elif result.get("result", {}).get("llm_analysis"):
                        # Use any agent's LLM analysis directly (generic handling)
                        analysis_content.append(result["result"]["llm_analysis"])
                        agent_name = result.get("agent", "Unknown")
                        logger.info(f"🤖 Using LLM insights from {agent_name}")
                    else:
                        # Keep other results for potential synthesis
                        non_analysis_results.append(result)

                if analysis_content and not non_analysis_results:
                    # If we only have analysis content, use it directly
                    final_answer = "\n\n".join(analysis_content)
                    logger.info(f"✅ Using analysis content directly ({len(final_answer)} characters)")
                elif analysis_content and non_analysis_results:
                    # Mix analysis content with synthesis of other results
                    prompt = f"""The user asked: "{query}"

I have some pre-formatted analysis and some additional tool results to combine:

ANALYSIS:
{chr(10).join(analysis_content)}

ADDITIONAL RESULTS:
{json.dumps(non_analysis_results, indent=2)}

Please provide a brief introduction and then present the analysis, followed by any additional insights from the other results. Keep the analysis sections intact."""

                    final_answer = self.llm.call(prompt)
                    logger.info(f"✅ Combined analysis with synthesis ({len(final_answer)} characters)")
                else:
                    # Traditional synthesis for cases without specialized analysis
                    prompt = f"""Based on these tool execution results, provide a comprehensive answer to: "{query}"

Results:
{json.dumps([r for r in results if r.get('agent') != 'VisualizationAgent'], indent=2)}

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

    def _find_previous_data_agent(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the most recent previous agent that produced tool_data (generic detection)"""
        # Search in reverse order to get the most recent data agent
        for prev_result in reversed(results):
            if (prev_result.get("success") and
                prev_result.get("result", {}).get("tool_data") and
                isinstance(prev_result["result"]["tool_data"], str) and
                len(prev_result["result"]["tool_data"].strip()) > 0):
                logger.info(f"🔍 Found data agent: {prev_result.get('agent', 'Unknown')} with tool_data")
                return prev_result
        return None

    def _export_tool_data_to_file(self, tool_data: str) -> str:
        """Export tool_data to file (pure file I/O, no data processing)"""
        try:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_file.write(tool_data)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            logger.error(f"🎯 Error exporting tool data to file: {str(e)}")
            return None

    def _generate_generic_visualization_query(self, user_query: str) -> str:
        """Generate generic visualization query based only on user's ask"""
        try:
            # Generic query generation based on user language only
            query_lower = user_query.lower()

            if any(keyword in query_lower for keyword in ["line chart", "trends", "over time", "time series"]):
                return "Create a line chart showing trends over time. Auto-detect appropriate columns for x-axis and y-axis"
            elif any(keyword in query_lower for keyword in ["bar chart", "comparison", "compare"]):
                return "Create a bar chart for comparison. Auto-detect appropriate columns for categories and values"
            elif any(keyword in query_lower for keyword in ["pie chart", "distribution", "breakdown"]):
                return "Create a pie chart showing distribution. Auto-detect appropriate columns for categories and values"
            elif any(keyword in query_lower for keyword in ["scatter", "correlation", "relationship"]):
                return "Create a scatter plot to show relationships. Auto-detect appropriate columns for x and y axes"
            else:
                # Default to smart auto-detection based on data
                return "Create an appropriate visualization based on the data structure. Auto-detect the best chart type and columns"

        except Exception as e:
            logger.error(f"🎯 Error generating generic visualization query: {str(e)}")
            return f"Create a visualization for: {user_query}"

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
                        if attached_file and agent_name in ["image_analysis", "data_processing", "presentation", "visualization", "cost_analysis"]:
                            file_path = attached_file.get("path", "")
                            logger.info(f"🔍 Debug attached_file: {attached_file}")
                            logger.info(f"🔍 Debug file_path: '{file_path}'")
                            if file_path:
                                agent_query += f" FILE_PATH:{file_path}"
                                logger.info(f"📎 Added file attachment: {file_path}")
                            else:
                                logger.warning(f"⚠️ No file path found in attached_file for {agent_name}")
                        logger.info(f"🎯 Delegating original query to {agent_name} specialist...")
                    else:
                        # Subsequent agents get context from previous results
                        logger.info(f"🔗 Building context-aware query for {agent_name} based on previous results...")

                        # Handle forecast tool when following any data-producing agent
                        if agent_name == "forecast":
                            # Find any previous agent that produced tool_data (generic detection)
                            data_agent = self._find_previous_data_agent(results)

                            if data_agent:
                                # Export tool_data to file (pure file I/O, no data processing)
                                file_path = self._export_tool_data_to_file(data_agent["result"]["tool_data"])

                                if file_path:
                                    agent_query = f"predict future trends using LSTM neural network FILE_PATH:{file_path}"
                                    logger.info(f"🎯 Exported data to file {file_path} for forecast")
                                else:
                                    agent_query = f"No data available for forecasting from previous agent"
                            else:
                                agent_query = f"predict future trends based on: {query}"
                        # Handle visualization tool when following any data-producing agent
                        elif agent_name == "visualization":
                            # Find any previous agent that produced tool_data (generic detection)
                            data_agent = self._find_previous_data_agent(results)

                            if data_agent:
                                # Export tool_data to file (pure file I/O, no data processing)
                                file_path = self._export_tool_data_to_file(data_agent["result"]["tool_data"])

                                if file_path:
                                    # Generate generic visualization query based on user's ask
                                    viz_query = self._generate_generic_visualization_query(query)
                                    agent_query = f"{viz_query} FILE_PATH:{file_path}"
                                    logger.info(f"🎯 Exported data to file {file_path} for visualization")
                                else:
                                    agent_query = f"No data available for visualization from previous agent"
                            else:
                                agent_query = f"Create a visualization for: {query}"
                        else:
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

                # Separate analysis results from visualization results
                analysis_content = []
                non_analysis_results = []

                for result in results:
                    if result.get("agent") == "VisualizationAgent":
                        # Visualization results are displayed separately, not included in final answer
                        logger.info(f"📊 Visualization result will be displayed separately")
                        continue
                    elif result.get("result", {}).get("llm_analysis"):
                        # Use any agent's LLM analysis directly (generic handling)
                        analysis_content.append(result["result"]["llm_analysis"])
                        agent_name = result.get("agent", "Unknown")
                        logger.info(f"🤖 Using LLM insights from {agent_name}")
                    else:
                        # Keep other results for potential synthesis
                        non_analysis_results.append(result)

                if analysis_content and not non_analysis_results:
                    # If we only have analysis content, use it directly
                    final_answer = "\n\n".join(analysis_content)
                    logger.info(f"✅ Using analysis content directly ({len(final_answer)} characters)")
                elif analysis_content and non_analysis_results:
                    # Mix analysis content with synthesis of other results
                    prompt = f"""The user asked: "{query}"

I have some pre-formatted analysis and some additional tool results to combine:

ANALYSIS:
{chr(10).join(analysis_content)}

ADDITIONAL RESULTS:
{json.dumps(non_analysis_results, indent=2)}

Please provide a brief introduction and then present the analysis, followed by any additional insights from the other results. Keep the analysis sections intact."""

                    final_answer = self.llm.call(prompt)
                    logger.info(f"✅ Combined analysis with synthesis ({len(final_answer)} characters)")
                else:
                    # Traditional synthesis for cases without specialized analysis
                    prompt = f"""Based on these tool execution results, provide a comprehensive answer to: "{query}"

Results:
{json.dumps([r for r in results if r.get('agent') != 'VisualizationAgent'], indent=2)}

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

    def _find_previous_data_agent(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the most recent previous agent that produced tool_data (generic detection)"""
        # Search in reverse order to get the most recent data agent
        for prev_result in reversed(results):
            if (prev_result.get("success") and
                prev_result.get("result", {}).get("tool_data") and
                isinstance(prev_result["result"]["tool_data"], str) and
                len(prev_result["result"]["tool_data"].strip()) > 0):
                logger.info(f"🔍 Found data agent: {prev_result.get('agent', 'Unknown')} with tool_data")
                return prev_result
        return None

    def _export_tool_data_to_file(self, tool_data: str) -> str:
        """Export tool_data to file (pure file I/O, no data processing)"""
        try:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_file.write(tool_data)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            logger.error(f"🎯 Error exporting tool data to file: {str(e)}")
            return None

    def _generate_generic_visualization_query(self, user_query: str) -> str:
        """Generate generic visualization query based only on user's ask"""
        try:
            # Generic query generation based on user language only
            query_lower = user_query.lower()

            if any(keyword in query_lower for keyword in ["line chart", "trends", "over time", "time series"]):
                return "Create a line chart showing trends over time. Auto-detect appropriate columns for x-axis and y-axis"
            elif any(keyword in query_lower for keyword in ["bar chart", "comparison", "compare"]):
                return "Create a bar chart for comparison. Auto-detect appropriate columns for categories and values"
            elif any(keyword in query_lower for keyword in ["pie chart", "distribution", "breakdown"]):
                return "Create a pie chart showing distribution. Auto-detect appropriate columns for categories and values"
            elif any(keyword in query_lower for keyword in ["scatter", "correlation", "relationship"]):
                return "Create a scatter plot to show relationships. Auto-detect appropriate columns for x and y axes"
            else:
                # Default to smart auto-detection based on data
                return "Create an appropriate visualization based on the data structure. Auto-detect the best chart type and columns"

        except Exception as e:
            logger.error(f"🎯 Error generating generic visualization query: {str(e)}")
            return f"Create a visualization for: {user_query}"