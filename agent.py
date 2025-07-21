# agent.py - Backend Agent with LLM Integration
import json
import re
import requests
from typing import Dict, Any, Optional
from a2a_protocol import A2AClient
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class CalculatorAgent:
    def __init__(self):
        # Initialize OpenAI client (you can also use other LLMs)
        self.llm_client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY', 'your-api-key-here')
        )
        
        # Initialize A2A client to communicate with MCP server
        self.a2a_client = A2AClient(mcp_server_url="http://localhost:8001")
        
        # System prompt for the LLM
        self.system_prompt = """
        You are a calculator assistant. Your job is to:
        1. Understand user requests for mathematical calculations
        2. Extract the mathematical expression from natural language
        3. Determine which calculator function to call
        4. Format the response in a user-friendly way
        
        Available calculator functions via MCP:
        - add(a, b): Addition
        - subtract(a, b): Subtraction  
        - multiply(a, b): Multiplication
        - divide(a, b): Division
        - calculate(expression): Evaluate complex mathematical expressions
        
        When you identify a calculation request, respond with JSON:
        {
            "intent": "calculate",
            "function": "function_name",
            "parameters": {"param1": value1, "param2": value2},
            "expression": "original_expression",
            "user_friendly_response": "natural language response"
        }
        
        For non-calculation requests, respond with:
        {
            "intent": "chat",
            "user_friendly_response": "your response"
        }
        """
    
    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Main processing pipeline:
        1. Use LLM to understand user intent and extract calculation
        2. If calculation needed, call MCP server via A2A protocol
        3. Format and return response
        """
        try:
            # Step 1: Use LLM to understand the message
            llm_response = self._analyze_with_llm(user_message)
            
            # Step 2: Parse LLM response
            parsed_response = self._parse_llm_response(llm_response)
            
            # Step 3: Handle based on intent
            if parsed_response.get('intent') == 'calculate':
                return self._handle_calculation(parsed_response, user_message)
            else:
                return {
                    'message': parsed_response.get('user_friendly_response', 
                                                  'I can help you with calculations. Try asking me to add, subtract, multiply, divide numbers or evaluate expressions!'),
                    'type': 'chat'
                }
        
        except Exception as e:
            return {
                'message': f'Sorry, I encountered an error: {str(e)}. Please try again.',
                'type': 'error'
            }
    
    def _analyze_with_llm(self, user_message: str) -> str:
        """Use LLM to analyze user message and extract calculation intent"""
        try:
            # For demo purposes, if no OpenAI key, use simple pattern matching
            if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your-api-key-here':
                return self._simple_pattern_matching(user_message)
            
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Fallback to pattern matching if LLM fails
            print(f"LLM failed, using fallback: {e}")
            return self._simple_pattern_matching(user_message)
    
    def _simple_pattern_matching(self, user_message: str) -> str:
        """Simple pattern matching fallback when LLM is not available"""
        message = user_message.lower()
        
        # Look for mathematical expressions
        math_patterns = [
            r'(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)',  # addition
            r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)',   # subtraction
            r'(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)',  # multiplication
            r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)',   # division
        ]
        
        operations = ['add', 'subtract', 'multiply', 'divide']
        
        # Check for explicit operations
        for op in operations:
            if op in message:
                numbers = re.findall(r'\d+(?:\.\d+)?', user_message)
                if len(numbers) >= 2:
                    return json.dumps({
                        "intent": "calculate",
                        "function": op,
                        "parameters": {"a": float(numbers[0]), "b": float(numbers[1])},
                        "expression": user_message,
                        "user_friendly_response": f"I'll {op} {numbers[0]} and {numbers[1]} for you."
                    })
        
        # Check for mathematical expressions
        for i, pattern in enumerate(math_patterns):
            match = re.search(pattern, user_message)
            if match:
                a, b = float(match.group(1)), float(match.group(2))
                op_names = ['add', 'subtract', 'multiply', 'divide']
                return json.dumps({
                    "intent": "calculate",
                    "function": op_names[i],
                    "parameters": {"a": a, "b": b},
                    "expression": user_message,
                    "user_friendly_response": f"I'll calculate {match.group(0)} for you."
                })
        
        # Check for complex expressions
        if re.search(r'[\d+\-*/().\s]+', user_message) and any(op in user_message for op in ['+', '-', '*', '/']):
            return json.dumps({
                "intent": "calculate",
                "function": "calculate",
                "parameters": {"expression": user_message},
                "expression": user_message,
                "user_friendly_response": f"I'll evaluate the expression: {user_message}"
            })
        
        return json.dumps({
            "intent": "chat",
            "user_friendly_response": "I can help you with calculations! Try asking me to add numbers like '5 + 3' or 'add 10 and 20'."
        })
    
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse the LLM response JSON"""
        try:
            # Try to extract JSON from the response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = llm_response[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(llm_response)
        
        except json.JSONDecodeError:
            # If not JSON, treat as chat response
            return {
                "intent": "chat",
                "user_friendly_response": llm_response
            }
    
    def _handle_calculation(self, parsed_response: Dict[str, Any], original_message: str) -> Dict[str, Any]:
        """Handle calculation by calling MCP server via A2A protocol"""
        try:
            function_name = parsed_response.get('function')
            parameters = parsed_response.get('parameters', {})
            expression = parsed_response.get('expression', original_message)
            
            # Call MCP server via A2A protocol
            result = self.a2a_client.call_mcp_function(function_name, parameters)
            
            if result.get('success'):
                calculation_result = result.get('result')
                return {
                    'message': f"The result is: {calculation_result}",
                    'calculation': expression,
                    'result': calculation_result,
                    'type': 'calculation'
                }
            else:
                error_msg = result.get('error', 'Unknown error occurred')
                return {
                    'message': f"Sorry, I couldn't perform the calculation: {error_msg}",
                    'type': 'error'
                }
        
        except Exception as e:
            return {
                'message': f"Error performing calculation: {str(e)}",
                'type': 'error'
            }