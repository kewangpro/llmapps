"""
MCP Client implementation
"""

# Standard library imports
import argparse
import asyncio
import json
import socket
import subprocess
import sys
from typing import Dict, Any, Optional, List, Tuple


class MCPError(Exception):
    """Base exception for MCP client errors"""
    pass


class ServerError(MCPError):
    """Exception raised when server returns an error"""
    pass


class ConfigError(MCPError):
    """Exception raised when there's a configuration error"""
    pass


class MCPClient:
    def __init__(self, config_path: str = "config.json"):
        """Initialize MCP client with configuration"""
        self.config = self._load_config(config_path)
        self.server_process = None
        self.server_url = f"http://localhost:{self.config.get('port', 8000)}"
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                "server": "mcp_server.py",
                "port": 8000,
                "mcpServers": {
                    "default": {
                        "command": ["python", "mcp_server.py"],
                        "url": "http://localhost:8000"
                    }
                }
            }
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON in config file: {e}")
    
    async def start_server(self) -> None:
        """Start the MCP server"""
        if self.server_process is not None:
            print("Server is already running")
            return
        
        try:
            # Get server command from config
            server_config = self.config.get("mcpServers", {}).get("default", {})
            command = server_config.get("command", ["python", "mcp_server.py"])
            
            # Start server process
            print(f"Starting server with command: {' '.join(command)}")
            self.server_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Wait for server to start
            max_retries = 10
            retry_delay = 1
            
            for i in range(max_retries):
                # Check if server process is still running
                if self.server_process.poll() is not None:
                    stderr = self.server_process.stderr.read()
                    raise ServerError(f"Server failed to start: {stderr}")
                
                # Try to connect to server
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', self.config.get('port', 8000)))
                    sock.close()
                    
                    if result == 0:
                        print("Server started successfully")
                        return
                except Exception as e:
                    print(f"Connection attempt {i+1} failed: {e}")
                
                await asyncio.sleep(retry_delay)
            
            raise ServerError("Server failed to start: Timeout waiting for server to be ready")
            
        except Exception as e:
            await self.cleanup()
            raise ServerError(f"Failed to start server: {e}")
    
    async def cleanup(self) -> None:
        """Clean up server process"""
        if self.server_process is not None:
            try:
                self.server_process.terminate()
                await asyncio.sleep(0.1)  # Give process time to terminate
                if self.server_process.poll() is None:
                    self.server_process.kill()
            finally:
                self.server_process = None
    
    async def send_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to MCP server"""
        if self.server_process is None:
            raise ServerError("Server is not running")
        
        try:
            # Prepare curl command
            url = f"{self.server_url}/{endpoint}"
            data_str = json.dumps(data)
            
            # Execute curl command
            curl_cmd = [
                "curl", "-s",
                "-X", "POST",
                "-H", "Content-Type: application/json",
                "-d", data_str,
                url
            ]
            
            process = await asyncio.create_subprocess_exec(
                *curl_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise ServerError(f"Request failed with code {process.returncode}")
            
            if not stdout:
                raise ServerError("Empty response from server")
            
            try:
                response = json.loads(stdout.decode())
            except json.JSONDecodeError as e:
                raise ServerError(f"Invalid JSON response: {e}")
            
            if "error" in response:
                raise ServerError(f"Server error: {response['error']}")
            
            return response
            
        except Exception as e:
            raise ServerError(f"Error sending request: {e}")
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize MCP server"""
        return await self.send_request("initialize", {})
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available tools"""
        return await self.send_request("list_tools", {})
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name with arguments"""
        return await self.send_request("call_tool", {
            "name": name,
            "arguments": arguments
        })


class MCPTester:
    def __init__(self, client: MCPClient):
        self.client = client
        self.test_results: List[Tuple[str, bool, str]] = []
    
    async def run_test(self, name: str, test_func: callable) -> None:
        """Run a single test"""
        try:
            print(f"\n[Test] Running test: {name}")
            await test_func()
            self.test_results.append((name, True, "Test passed"))
            print(f"[Test] ✓ {name} passed")
        except Exception as e:
            self.test_results.append((name, False, str(e)))
            print(f"[Test] ✗ {name} failed: {e}")
    
    async def test_initialize(self) -> None:
        """Test server initialization"""
        print("[Test] Testing server initialization...")
        response = await self.client.initialize()
        print(f"[Test] Initialize response: {json.dumps(response, indent=2)}")
        
        if not isinstance(response, dict):
            raise AssertionError("Invalid response type")
        if "server" not in response:
            raise AssertionError("Missing server info in response")
        if "tools" not in response:
            raise AssertionError("Missing tools list in response")
        print("[Test] Initialize test passed all assertions")
    
    async def test_tools_list(self) -> None:
        """Test tools list endpoint"""
        print("[Test] Testing tools list endpoint...")
        response = await self.client.list_tools()
        print(f"[Test] Tools list response: {json.dumps(response, indent=2)}")
        
        if not isinstance(response, dict):
            raise AssertionError("Invalid response type")
        if "tools" not in response:
            raise AssertionError("Missing tools list in response")
        if not isinstance(response["tools"], list):
            raise AssertionError("Tools should be a list")
        print("[Test] Tools list test passed all assertions")
    
    async def test_echo_tool(self) -> None:
        """Test echo tool"""
        print("[Test] Testing echo tool...")
        test_message = "Hello, MCP!"
        print(f"[Test] Sending test message: {test_message}")
        
        response = await self.client.call_tool("echo", {"message": test_message})
        print(f"[Test] Echo response: {json.dumps(response, indent=2)}")
        
        if not isinstance(response, dict):
            raise AssertionError("Invalid response type")
        if "result" not in response:
            raise AssertionError("Missing result in response")
        if "content" not in response["result"]:
            raise AssertionError("Missing content in result")
        if not response["result"]["content"]:
            raise AssertionError("Empty content in result")
        if response["result"]["content"][0]["text"] != test_message:
            raise AssertionError("Echo message doesn't match input")
        print("[Test] Echo test passed all assertions")
    
    async def test_get_time_tool(self) -> None:
        """Test get_time tool"""
        print("[Test] Testing get_time tool...")
        response = await self.client.call_tool("get_time", {})
        print(f"[Test] Get time response: {json.dumps(response, indent=2)}")
        
        if not isinstance(response, dict):
            raise AssertionError("Invalid response type")
        if "result" not in response:
            raise AssertionError("Missing result in response")
        if "content" not in response["result"]:
            raise AssertionError("Missing content in result")
        if not response["result"]["content"]:
            raise AssertionError("Empty content in result")
        
        # Parse time info
        time_info = json.loads(response["result"]["content"][0]["text"])
        print(f"[Test] Parsed time info: {json.dumps(time_info, indent=2)}")
        
        required_fields = ["timestamp", "formatted", "timezone", "components"]
        for field in required_fields:
            if field not in time_info:
                raise AssertionError(f"Missing {field} in time info")
        print("[Test] Get time test passed all assertions")
    
    async def test_search_web_tool(self) -> None:
        """Test search_web tool"""
        print("[Test] Testing search_web tool...")
        test_query = "Python programming"
        print(f"[Test] Sending test query: {test_query}")
        
        response = await self.client.call_tool("search_web", {"query": test_query})
        print(f"[Test] Search web response: {json.dumps(response, indent=2)}")
        
        if not isinstance(response, dict):
            raise AssertionError("Invalid response type")
        if "result" not in response:
            raise AssertionError("Missing result in response")
        if "content" not in response["result"]:
            raise AssertionError("Missing content in result")
        if not response["result"]["content"]:
            raise AssertionError("Empty content in result")
        
        # Parse search results
        search_results = json.loads(response["result"]["content"][0]["text"])
        print(f"[Test] Parsed search results: {json.dumps(search_results, indent=2)}")
        
        if "query" not in search_results:
            raise AssertionError("Missing query in search results")
        if "results" not in search_results:
            raise AssertionError("Missing results in search results")
        if not isinstance(search_results["results"], list):
            raise AssertionError("Results should be a list")
        if not search_results["results"]:
            raise AssertionError("Empty results list")
        
        # Validate first result
        first_result = search_results["results"][0]
        print(f"[Test] First search result: {json.dumps(first_result, indent=2)}")
        
        required_fields = ["title", "content", "source"]
        for field in required_fields:
            if field not in first_result:
                raise AssertionError(f"Missing {field} in search result")
        print("[Test] Search web test passed all assertions")
    
    async def run_all_tests(self) -> None:
        """Run all tests"""
        try:
            print("\n[Test] Starting test suite...")
            # Start server
            print("[Test] Starting server...")
            await self.client.start_server()
            print("[Test] Server started successfully")
            
            # Run tests
            await self.run_test("Initialize", self.test_initialize)
            await self.run_test("Tools List", self.test_tools_list)
            await self.run_test("Echo Tool", self.test_echo_tool)
            await self.run_test("Get Time Tool", self.test_get_time_tool)
            await self.run_test("Search Web Tool", self.test_search_web_tool)
            
            # Print summary
            print("\n[Test] Test Summary:")
            print("=" * 50)
            total = len(self.test_results)
            passed = sum(1 for _, success, _ in self.test_results if success)
            failed = total - passed
            
            print(f"[Test] Total tests: {total}")
            print(f"[Test] Passed: {passed}")
            print(f"[Test] Failed: {failed}")
            print("\n[Test] Detailed Results:")
            print("-" * 50)
            
            for name, success, message in self.test_results:
                status = "✓" if success else "✗"
                print(f"[Test] {status} {name}: {message}")
            
            if failed > 0:
                sys.exit(1)
            
        finally:
            print("\n[Test] Cleaning up...")
            await self.client.cleanup()
            print("[Test] Cleanup complete")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MCP Client")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--server", help="Server script path", default="mcp_server.py")
    args = parser.parse_args()
    
    # Create client
    client = MCPClient()
    
    if args.test:
        # Run tests
        tester = MCPTester(client)
        await tester.run_all_tests()
    else:
        # Interactive mode
        try:
            await client.start_server()
            print("\nMCP Client Interactive Mode")
            print("Available commands:")
            print("  list_tools - List available tools")
            print("  call <tool> <json_args> - Call a tool with arguments")
            print("  exit - Exit the client")
            
            while True:
                try:
                    cmd = input("\nEnter command: ").strip()
                    
                    if cmd == "exit":
                        break
                    elif cmd == "list_tools":
                        response = await client.list_tools()
                        print(json.dumps(response, indent=2))
                    elif cmd.startswith("call "):
                        parts = cmd[5:].strip().split(" ", 1)
                        if len(parts) != 2:
                            print("Invalid command format. Use: call <tool> <json_args>")
                            continue
                        
                        tool_name, args_str = parts
                        try:
                            arguments = json.loads(args_str)
                            response = await client.call_tool(tool_name, arguments)
                            print(json.dumps(response, indent=2))
                        except json.JSONDecodeError:
                            print("Invalid JSON arguments")
                    else:
                        print("Unknown command")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
        
        finally:
            await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())