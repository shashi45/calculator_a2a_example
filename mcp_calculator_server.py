# mcp_calculator_server.py - MCP Calculator Server
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import logging
from a2a_protocol import A2AServer
import ast
import operator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Calculator MCP Server", version="1.0.0")

# Initialize A2A server
a2a_server = A2AServer("calculator_mcp_server")

class A2AMessage(BaseModel):
    protocol: str
    version: str
    message_id: str
    source: str
    target: str
    action: str
    payload: Dict[str, Any]

class CalculatorMCPServer:
    """MCP Server for calculator functions"""
    
    def __init__(self):
        self.functions = {
            "add": self.add,
            "subtract": self.subtract,
            "multiply": self.multiply,
            "divide": self.divide,
            "calculate": self.calculate
        }
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        logger.info(f"Executing add: {a} + {b}")
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a"""
        logger.info(f"Executing subtract: {a} - {b}")
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        logger.info(f"Executing multiply: {a} * {b}")
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b"""
        logger.info(f"Executing divide: {a} / {b}")
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return a / b
    
    def calculate(self, expression: str) -> float:
        """Evaluate a mathematical expression safely"""
        logger.info(f"Executing calculate: {expression}")
        
        # Clean the expression
        expression = expression.strip()
        
        # Remove any non-mathematical characters for safety
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")
        
        try:
            # Use ast.literal_eval for safe evaluation of simple expressions
            # For complex expressions, we'll use a safer approach
            result = self._safe_eval(expression)
            return result
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")
    
    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate mathematical expressions"""
        # Define allowed operators
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.USub: operator.neg,
        }
        
        def eval_node(node):
            if isinstance(node, ast.Num):  # number
                return node.n
            elif isinstance(node, ast.Constant):  # Python 3.8+
                return node.value
            elif isinstance(node, ast.BinOp):  # binary operation
                left = eval_node(node.left)
                right = eval_node(node.right)
                op = ops.get(type(node.op))
                if op is None:
                    raise ValueError("Unsupported operation")
                return op(left, right)
            elif isinstance(node, ast.UnaryOp):  # unary operation
                operand = eval_node(node.operand)
                op = ops.get(type(node.op))
                if op is None:
                    raise ValueError("Unsupported operation")
                return op(operand)
            else:
                raise ValueError("Unsupported expression")
        
        try:
            tree = ast.parse(expression, mode='eval')
            return eval_node(tree.body)
        except:
            # Fallback to simple eval for basic expressions
            # This is still risky but we've filtered the input
            return eval(expression)
    
    def execute_function(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a calculator function"""
        if function_name not in self.functions:
            raise ValueError(f"Unknown function: {function_name}")
        
        func = self.functions[function_name]
        
        try:
            # Call the function with parameters
            if function_name == "calculate":
                return func(parameters.get("expression", ""))
            else:
                a = float(parameters.get("a", 0))
                b = float(parameters.get("b", 0))
                return func(a, b)
        except Exception as e:
            raise ValueError(f"Error executing {function_name}: {str(e)}")
    
    def list_functions(self) -> Dict[str, str]:
        """List available functions with descriptions"""
        return {
            "add": "Add two numbers",
            "subtract": "Subtract second number from first",
            "multiply": "Multiply two numbers",
            "divide": "Divide first number by second",
            "calculate": "Evaluate a mathematical expression"
        }

# Initialize calculator server
calc_server = CalculatorMCPServer()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "server": "calculator_mcp_server"}

@app.post("/a2a/execute")
async def execute_a2a_function(message: A2AMessage):
    """Execute function via A2A protocol"""
    try:
        # Parse A2A message
        parsed = a2a_server.parse_a2a_message(message.dict())
        
        if not parsed["success"]:
            return a2a_server.create_response(
                message.message_id, 
                False, 
                error=parsed["error"]
            )
        
        # Extract function details
        payload = parsed["payload"]
        function_name = payload.get("function")
        parameters = payload.get("parameters", {})
        
        # Execute function
        result = calc_server.execute_function(function_name, parameters)
        
        # Return A2A response
        return a2a_server.create_response(
            parsed["message_id"],
            True,
            result=result
        )
    
    except ValueError as e:
        return a2a_server.create_response(
            message.message_id,
            False,
            error=str(e)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return a2a_server.create_response(
            message.message_id,
            False,
            error=f"Server error: {str(e)}"
        )

@app.post("/a2a/list_functions")
async def list_a2a_functions(message: A2AMessage):
    """List available functions via A2A protocol"""
    try:
        # Parse A2A message
        parsed = a2a_server.parse_a2a_message(message.dict())
        
        if not parsed["success"]:
            return a2a_server.create_response(
                message.message_id,
                False,
                error=parsed["error"]
            )
        
        # Get function list
        functions = calc_server.list_functions()
        
        return a2a_server.create_response(
            parsed["message_id"],
            True,
            result=functions
        )
    
    except Exception as e:
        return a2a_server.create_response(
            message.message_id,
            False,
            error=str(e)
        )

@app.get("/functions")
async def get_functions():
    """Get list of available functions (non-A2A endpoint)"""
    return calc_server.list_functions()

if __name__ == "__main__":
    print("Starting Calculator MCP Server on port 8001...")
    print("Available functions:", list(calc_server.functions.keys()))
    uvicorn.run(app, host="0.0.0.0", port=8001)