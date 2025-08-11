#!/usr/bin/env python3
"""
Corrected implementation using proper FastMCP and MCP Python SDK
"""

import os
import yaml
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import logging

# Correct FastMCP import - only for creating servers
from fastmcp import FastMCP

# For MCP client functionality, use the official MCP Python SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

# ====================================================================
# BACKSTAGE TEMPLATE DEFINITIONS AND RAG IMPLEMENTATION (same as original)
# ====================================================================

@dataclass
class BackstageTemplate:
    """Represents a Backstage software template"""
    name: str
    template_path: Path
    yaml_content: Dict[str, Any]
    description: str
    parameters: Dict[str, Any]
    steps: List[Dict[str, Any]]
    output: Dict[str, Any]
    
    @classmethod
    def from_yaml_file(cls, template_path: Path) -> 'BackstageTemplate':
        """Load template from template.yaml file"""
        with open(template_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
        
        spec = yaml_content.get('spec', {})
        return cls(
            name=yaml_content.get('metadata', {}).get('name', template_path.stem),
            template_path=template_path,
            yaml_content=yaml_content,
            description=yaml_content.get('metadata', {}).get('description', ''),
            parameters=spec.get('parameters', {}),
            steps=spec.get('steps', []),
            output=spec.get('output', {})
        )

class TemplateRAGManager:
    """Manages Backstage templates as RAG context for LLM"""
    
    def __init__(self, templates_dir: str = "backstage_templates"):
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, BackstageTemplate] = {}
        self.vector_store: Optional[FAISS] = None
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
    async def initialize(self):
        """Initialize RAG system with template content"""
        await self._load_templates()
        await self._build_vector_store()
    
    async def _load_templates(self):
        """Load all template.yaml files from templates directory"""
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            await self._create_sample_templates()
        
        for template_file in self.templates_dir.rglob("template.yaml"):
            try:
                template = BackstageTemplate.from_yaml_file(template_file)
                self.templates[template.name] = template
                logger.info(f"Loaded template: {template.name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
    
    async def _create_sample_templates(self):
        """Create sample Backstage templates for demonstration"""
        # Same implementation as original...
        lambda_template = {
            "apiVersion": "scaffolder.backstage.io/v1beta3",
            "kind": "Template",
            "metadata": {
                "name": "lambda-template",
                "title": "AWS Lambda Function",
                "description": "Create a new AWS Lambda function with best practices",
                "tags": ["aws", "lambda", "serverless", "python"]
            },
            "spec": {
                "owner": "platform-team",
                "type": "service",
                "parameters": [
                    {
                        "title": "Component Information",
                        "required": ["component_name", "description"],
                        "properties": {
                            "component_name": {
                                "title": "Component Name",
                                "type": "string",
                                "description": "Unique name for the Lambda function",
                                "pattern": "^[a-z][a-z0-9-]*[a-z0-9]$",
                                "ui:autofocus": True
                            },
                            "description": {
                                "title": "Description",
                                "type": "string",
                                "description": "A brief description of what this Lambda does"
                            }
                        }
                    }
                ]
            }
        }
        
        # Save template
        lambda_dir = self.templates_dir / "lambda-template"
        lambda_dir.mkdir(parents=True, exist_ok=True)
        with open(lambda_dir / "template.yaml", 'w') as f:
            yaml.dump(lambda_template, f, default_flow_style=False, sort_keys=False)
    
    async def _build_vector_store(self):
        """Build vector store from template content for RAG"""
        documents = []
        
        for template_name, template in self.templates.items():
            # Create comprehensive template documentation
            template_doc = self._template_to_document(template)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(template_doc)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "template_name": template_name,
                        "chunk_id": i,
                        "template_type": template.yaml_content.get("spec", {}).get("type", "unknown"),
                        "tags": template.yaml_content.get("metadata", {}).get("tags", [])
                    }
                )
                documents.append(doc)
        
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Built vector store with {len(documents)} chunks from {len(self.templates)} templates")
    
    def _template_to_document(self, template: BackstageTemplate) -> str:
        """Convert template to comprehensive text document for RAG"""
        doc_parts = [
            f"Template Name: {template.name}",
            f"Description: {template.description}",
            f"Type: {template.yaml_content.get('spec', {}).get('type', 'unknown')}",
            f"Tags: {', '.join(template.yaml_content.get('metadata', {}).get('tags', []))}",
        ]
        return "\n".join(doc_parts)
    
    async def get_template_context(self, template_type: str, user_query: str) -> Dict[str, Any]:
        """Get relevant template context using RAG"""
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        search_query = f"{template_type} {user_query}"
        relevant_docs = self.vector_store.similarity_search(search_query, k=3)
        
        matching_template = None
        for template_name, template in self.templates.items():
            if template_type.lower() in template_name.lower():
                matching_template = template
                break
        
        context = {
            "template_found": matching_template is not None,
            "relevant_content": [doc.page_content for doc in relevant_docs],
            "template_metadata": {}
        }
        
        if matching_template:
            context["template_metadata"] = {
                "name": matching_template.name,
                "description": matching_template.description,
                "parameters": matching_template.parameters,
            }
        
        return context

# ====================================================================
# CORRECTED MCP CLIENT IMPLEMENTATION
# ====================================================================

@dataclass
class MCPServerInfo:
    """Information about an MCP server"""
    name: str
    command: str
    args: List[str]
    status: str = "unknown"
    capabilities: List[str] = field(default_factory=list)

class CorrectMCPClient:
    """Proper MCP client implementation using official Python SDK"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerInfo] = {}
        self.active_sessions: Dict[str, ClientSession] = {}
    
    def register_server(self, name: str, command: str, args: List[str] = None):
        """Register an MCP server"""
        self.servers[name] = MCPServerInfo(
            name=name,
            command=command,
            args=args or [],
        )
    
    async def connect_to_server(self, server_name: str) -> bool:
        """Connect to an MCP server using stdio"""
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not registered")
            return False
        
        server_info = self.servers[server_name]
        
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=server_info.command,
                args=server_info.args,
            )
            
            # Create and initialize session
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the connection
                    await session.initialize()
                    
                    # Store session
                    self.active_sessions[server_name] = session
                    server_info.status = "connected"
                    
                    # Get available tools
                    result = await session.list_tools()
                    server_info.capabilities = [tool.name for tool in result.tools] if result.tools else []
                    
                    logger.info(f"Connected to {server_name} with tools: {server_info.capabilities}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            server_info.status = "failed"
            return False
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on an MCP server"""
        if server_name not in self.active_sessions:
            if not await self.connect_to_server(server_name):
                raise ValueError(f"Cannot connect to server {server_name}")
        
        session = self.active_sessions[server_name]
        
        try:
            result = await session.call_tool(tool_name, arguments)
            if result.content:
                # Parse the result content
                if len(result.content) > 0:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        try:
                            return json.loads(content.text)
                        except json.JSONDecodeError:
                            return {"result": content.text}
                    else:
                        return {"result": str(content)}
            return {"result": "No content returned"}
            
        except Exception as e:
            logger.error(f"Tool call failed {server_name}.{tool_name}: {e}")
            raise

# ====================================================================
# SIMPLIFIED ORCHESTRATOR
# ====================================================================

class SimplifiedOrchestrator:
    """Simplified orchestrator without the complex MCP discovery"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=0.1
        )
        
        self.template_rag = TemplateRAGManager()
        self.mcp_client = CorrectMCPClient()
        self.sessions = {}
        
    async def initialize(self):
        """Initialize the orchestrator"""
        logger.info("Initializing Simplified Orchestrator...")
        
        # Initialize template RAG
        await self.template_rag.initialize()
        logger.info(f"Loaded {len(self.template_rag.templates)} templates")
        
        # Register MCP servers (if they exist)
        self._register_mcp_servers()
        
        logger.info("Simplified Orchestrator initialized successfully")
    
    def _register_mcp_servers(self):
        """Register available MCP servers"""
        # Example server registrations
        # These would be actual MCP servers you have running
        
        # Example: Register a template server (if it exists)
        template_server_cmd = os.getenv("TEMPLATE_SERVER_CMD")
        if template_server_cmd:
            self.mcp_client.register_server(
                "template_server",
                template_server_cmd,
                []
            )
        
        # For development/testing, we can work without MCP servers
        logger.info("MCP server registration completed")
    
    async def process_user_request(self, user_input: str, session_id: str = "default") -> str:
        """Process user request with RAG context"""
        
        # Simple state management
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "state": "initial",
                "user_request": "",
                "template_type": "",
                "parameters": {}
            }
        
        session = self.sessions[session_id]
        
        if session["state"] == "initial":
            return await self._analyze_request(user_input, session)
        elif session["state"] == "gathering_info":
            return await self._gather_additional_info(user_input, session)
        else:
            return "Processing your request..."
    
    async def _analyze_request(self, user_input: str, session: Dict[str, Any]) -> str:
        """Analyze user request and determine template type"""
        
        # Update session
        session["user_request"] = user_input
        session["state"] = "gathering_info"
        
        # Use LLM to analyze request
        analysis_prompt = f"""
        Analyze this user request to determine what type of component they want to create:
        
        User request: "{user_input}"
        
        Available templates: {list(self.template_rag.templates.keys())}
        
        Respond with JSON:
        {{
            "template_type": "best_matching_template",
            "confidence": 0.0-1.0,
            "reasoning": "explanation",
            "likely_parameters": {{"param": "value"}}
        }}
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert at analyzing development requests."),
            HumanMessage(content=analysis_prompt)
        ])
        
        try:
            analysis = json.loads(response.content)
            template_type = analysis.get("template_type", "lambda-template")
        except:
            template_type = "lambda-template"
        
        session["template_type"] = template_type
        
        # Get template context from RAG
        context = await self.template_rag.get_template_context(template_type, user_input)
        
        # Build response
        if context["template_found"]:
            template_meta = context["template_metadata"]
            return f"""✅ **Analysis Complete!**

**Template Selected**: {template_meta['name']}
**Description**: {template_meta['description']}

**Based on your request**: "{user_input}"

I can help you create this component. What additional details would you like to provide?

Type 'proceed' to continue with default values, or provide specific information."""
        else:
            return f"""I'll help you create a {template_type} component based on your request: "{user_input}"

What specific details would you like to provide?

Type 'proceed' to continue."""
    
    async def _gather_additional_info(self, user_input: str, session: Dict[str, Any]) -> str:
        """Gather additional information or proceed with creation"""
        
        if user_input.lower().strip() == "proceed":
            return await self._create_component(session)
        
        # Parse additional information
        session["additional_info"] = user_input
        
        return f"""Thanks for the additional information: "{user_input}"

Type 'proceed' to create the component, or provide more details."""
    
    async def _create_component(self, session: Dict[str, Any]) -> str:
        """Create the component (simulation)"""
        
        template_type = session.get("template_type", "unknown")
        user_request = session.get("user_request", "")
        
        # Reset session
        session["state"] = "completed"
        
        return f"""🎉 **Component Created Successfully!**

**Template**: {template_type}
**Original Request**: {user_request}

**What would happen in production:**
• Generate component files from template
• Create repository
• Register in Backstage catalog
• Set up CI/CD pipeline

*This is a simulation - integrate with actual Backstage API for real component creation.*"""

# ====================================================================
# CORRECT FASTMCP SERVER IMPLEMENTATION
# ====================================================================

# Create a proper FastMCP server
template_server = FastMCP("TemplateServer")

@template_server.tool()
def get_template_info(template_name: str) -> dict:
    """Get information about a specific template"""
    # This would integrate with your actual template system
    templates = {
        "lambda-template": {
            "name": "AWS Lambda Function",
            "description": "Create a serverless Lambda function",
            "parameters": ["component_name", "runtime", "description"]
        },
        "microservice-template": {
            "name": "Microservice",
            "description": "Create a containerized microservice",
            "parameters": ["service_name", "port", "database_type"]
        }
    }
    
    return templates.get(template_name, {"error": "Template