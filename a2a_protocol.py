# a2a_protocol.py - A2A Protocol Implementation
import json
import requests
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class A2AClient:
    """
    A2A (Application-to-Application) Protocol Client
    This handles communication between the agent and MCP servers
    """
    
    def __init__(self, mcp_server_url: str):
        self.mcp_server_url = mcp_server_url
        self.session = requests.Session()
        
        # A2A Protocol Headers
        self.default_headers = {
            'Content-Type': 'application/json',
            'X-A2A-Protocol-Version': '1.0',
            'X-A2A-Client': 'calculator-agent'
        }
    
    def call_mcp_function(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a function on the MCP server using A2A protocol
        
        A2A Message Format:
        {
            "protocol": "a2a",
            "version": "1.0",
            "message_id": "unique_id",
            "source": "agent",
            "target": "mcp_server",
            "action": "function_call",
            "payload": {
                "function": "function_name",
                "parameters": {...}
            }
        }
        """
        try:
            # Create A2A protocol message
            a2a_message = {
                "protocol": "a2a",
                "version": "1.0",
                "message_id": self._generate_message_id(),
                "source": "calculator_agent",
                "target": "calculator_mcp_server",
                "action": "function_call",
                "payload": {
                    "function": function_name,
                    "parameters": parameters
                }
            }
            
            logger.info(f"Sending A2A message to MCP server: {function_name}")
            
            # Send request to MCP server
            response = self.session.post(
                f"{self.mcp_server_url}/a2a/execute",
                headers=self.default_headers,
                json=a2a_message,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Received response from MCP server: {result}")
                return result
            else:
                logger.error(f"MCP server error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"MCP server returned status {response.status_code}: {response.text}"
                }
        
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to MCP server")
            return {
                "success": False,
                "error": "Could not connect to MCP server. Make sure it's running on port 8001."
            }
        
        except requests.exceptions.Timeout:
            logger.error("Request to MCP server timed out")
            return {
                "success": False,
                "error": "Request to MCP server timed out"
            }
        
        except Exception as e:
            logger.error(f"Error calling MCP server: {str(e)}")
            return {
                "success": False,
                "error": f"Error calling MCP server: {str(e)}"
            }
    
    def health_check(self) -> bool:
        """Check if MCP server is healthy"""
        try:
            response = self.session.get(
                f"{self.mcp_server_url}/health",
                headers=self.default_headers,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def list_functions(self) -> Dict[str, Any]:
        """Get list of available functions from MCP server"""
        try:
            a2a_message = {
                "protocol": "a2a",
                "version": "1.0",
                "message_id": self._generate_message_id(),
                "source": "calculator_agent",
                "target": "calculator_mcp_server",
                "action": "list_functions",
                "payload": {}
            }
            
            response = self.session.post(
                f"{self.mcp_server_url}/a2a/list_functions",
                headers=self.default_headers,
                json=a2a_message,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": "Failed to list functions"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID for A2A protocol"""
        import uuid
        return str(uuid.uuid4())

class A2AServer:
    """
    A2A Protocol Server - to be used by MCP servers
    This handles incoming A2A protocol messages
    """
    
    def __init__(self, server_name: str):
        self.server_name = server_name
    
    def parse_a2a_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate A2A protocol message"""
        try:
            # Validate required fields
            required_fields = ["protocol", "version", "message_id", "source", "target", "action", "payload"]
            
            for field in required_fields:
                if field not in message:
                    return {
                        "success": False,
                        "error": f"Missing required field: {field}"
                    }
            
            # Validate protocol
            if message["protocol"] != "a2a":
                return {
                    "success": False,
                    "error": "Invalid protocol. Expected 'a2a'"
                }
            
            # Validate version
            if message["version"] != "1.0":
                return {
                    "success": False,
                    "error": "Unsupported protocol version"
                }
            
            return {
                "success": True,
                "action": message["action"],
                "payload": message["payload"],
                "message_id": message["message_id"]
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Error parsing A2A message: {str(e)}"
            }
    
    def create_response(self, message_id: str, success: bool, result: Any = None, error: str = None) -> Dict[str, Any]:
        """Create A2A protocol response"""
        response = {
            "protocol": "a2a",
            "version": "1.0",
            "message_id": message_id,
            "source": self.server_name,
            "response_to": message_id,
            "success": success
        }
        
        if success and result is not None:
            response["result"] = result
        elif not success and error:
            response["error"] = error
        
        return response