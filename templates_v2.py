#!/usr/bin/env python3
"""
Enhanced implementation with:
1. Real Backstage template.yaml files as RAG context
2. Dynamic MCP server discovery and tool calling
3. Template content parsing and LLM context injection
"""

import os
import yaml
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import logging

from fastmcp import FastMCP
from fastmcp.client import ClientFactory, MCPClient
from fastmcp.server import Server
from fastmcp.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

# ====================================================================
# BACKSTAGE TEMPLATE DEFINITIONS AND RAG IMPLEMENTATION
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
        
        # Lambda template
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
                            },
                            "owner_team": {
                                "title": "Owner Team",
                                "type": "string",
                                "description": "Team responsible for this component",
                                "default": "platform-team",
                                "enum": ["platform-team", "backend-team", "data-team"]
                            }
                        }
                    },
                    {
                        "title": "Lambda Configuration",
                        "properties": {
                            "runtime": {
                                "title": "Runtime",
                                "type": "string",
                                "description": "Lambda runtime environment",
                                "default": "python3.11",
                                "enum": ["python3.9", "python3.11", "nodejs18.x", "nodejs20.x"]
                            },
                            "memory_size": {
                                "title": "Memory Size (MB)",
                                "type": "integer",
                                "description": "Memory allocated to Lambda function",
                                "default": 128,
                                "minimum": 128,
                                "maximum": 10240
                            },
                            "timeout": {
                                "title": "Timeout (seconds)",
                                "type": "integer",
                                "description": "Maximum execution time",
                                "default": 30,
                                "minimum": 1,
                                "maximum": 900
                            },
                            "environment_variables": {
                                "title": "Environment Variables",
                                "type": "object",
                                "description": "Environment variables for the Lambda",
                                "default": {}
                            }
                        }
                    }
                ],
                "steps": [
                    {
                        "id": "fetch-base",
                        "name": "Fetch Base Template",
                        "action": "fetch:template",
                        "input": {
                            "url": "./skeleton",
                            "values": {
                                "component_name": "${{ parameters.component_name }}",
                                "description": "${{ parameters.description }}",
                                "owner_team": "${{ parameters.owner_team }}",
                                "runtime": "${{ parameters.runtime }}",
                                "memory_size": "${{ parameters.memory_size }}",
                                "timeout": "${{ parameters.timeout }}"
                            }
                        }
                    },
                    {
                        "id": "publish",
                        "name": "Publish to Repository",
                        "action": "publish:github",
                        "input": {
                            "repoUrl": "github.com?owner=my-org&repo=${{ parameters.component_name }}",
                            "description": "${{ parameters.description }}"
                        }
                    },
                    {
                        "id": "register",
                        "name": "Register in Catalog",
                        "action": "catalog:register",
                        "input": {
                            "repoContentsUrl": "${{ steps.publish.output.repoContentsUrl }}",
                            "catalogInfoPath": "/catalog-info.yaml"
                        }
                    }
                ],
                "output": {
                    "links": [
                        {
                            "title": "Repository",
                            "url": "${{ steps.publish.output.remoteUrl }}"
                        },
                        {
                            "title": "Catalog Entry",
                            "url": "${{ steps.register.output.catalogInfoUrl }}"
                        }
                    ]
                }
            }
        }
        
        # Microservice template
        microservice_template = {
            "apiVersion": "scaffolder.backstage.io/v1beta3",
            "kind": "Template",
            "metadata": {
                "name": "microservice-template",
                "title": "Microservice Application",
                "description": "Create a new microservice with FastAPI and Docker",
                "tags": ["microservice", "fastapi", "docker", "kubernetes"]
            },
            "spec": {
                "owner": "platform-team",
                "type": "service",
                "parameters": [
                    {
                        "title": "Service Information",
                        "required": ["service_name", "description", "port"],
                        "properties": {
                            "service_name": {
                                "title": "Service Name",
                                "type": "string",
                                "description": "Name of the microservice",
                                "pattern": "^[a-z][a-z0-9-]*[a-z0-9]$"
                            },
                            "description": {
                                "title": "Description",
                                "type": "string",
                                "description": "What does this service do?"
                            },
                            "port": {
                                "title": "Service Port",
                                "type": "integer",
                                "description": "Port the service listens on",
                                "default": 8000,
                                "minimum": 3000,
                                "maximum": 9999
                            },
                            "owner_team": {
                                "title": "Owner Team",
                                "type": "string",
                                "description": "Team responsible for this service",
                                "default": "backend-team",
                                "enum": ["platform-team", "backend-team", "data-team"]
                            }
                        }
                    },
                    {
                        "title": "Configuration",
                        "properties": {
                            "database_required": {
                                "title": "Requires Database",
                                "type": "boolean",
                                "description": "Does this service need a database?",
                                "default": False
                            },
                            "database_type": {
                                "title": "Database Type",
                                "type": "string",
                                "description": "Type of database",
                                "default": "postgresql",
                                "enum": ["postgresql", "mysql", "mongodb"],
                                "ui:widget": "radio",
                                "ui:options": {
                                    "inline": True
                                }
                            }
                        },
                        "dependencies": {
                            "database_required": {
                                "oneOf": [
                                    {
                                        "properties": {
                                            "database_required": {"const": False}
                                        }
                                    },
                                    {
                                        "properties": {
                                            "database_required": {"const": True}
                                        },
                                        "required": ["database_type"]
                                    }
                                ]
                            }
                        }
                    }
                ],
                "steps": [
                    {
                        "id": "fetch-base",
                        "name": "Fetch Base Template",
                        "action": "fetch:template",
                        "input": {
                            "url": "./microservice-skeleton",
                            "values": {
                                "service_name": "${{ parameters.service_name }}",
                                "description": "${{ parameters.description }}",
                                "port": "${{ parameters.port }}",
                                "owner_team": "${{ parameters.owner_team }}",
                                "database_required": "${{ parameters.database_required }}",
                                "database_type": "${{ parameters.database_type }}"
                            }
                        }
                    }
                ]
            }
        }
        
        # Save templates
        lambda_dir = self.templates_dir / "lambda-template"
        lambda_dir.mkdir(parents=True, exist_ok=True)
        with open(lambda_dir / "template.yaml", 'w') as f:
            yaml.dump(lambda_template, f, default_flow_style=False, sort_keys=False)
        
        microservice_dir = self.templates_dir / "microservice-template"
        microservice_dir.mkdir(parents=True, exist_ok=True)
        with open(microservice_dir / "template.yaml", 'w') as f:
            yaml.dump(microservice_template, f, default_flow_style=False, sort_keys=False)
    
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
            f"Owner: {template.yaml_content.get('spec', {}).get('owner', 'unknown')}",
            f"Tags: {', '.join(template.yaml_content.get('metadata', {}).get('tags', []))}",
            "",
            "Parameters:"
        ]
        
        # Add parameter details
        parameters = template.parameters
        if isinstance(parameters, list):
            for param_group in parameters:
                if 'properties' in param_group:
                    for param_name, param_def in param_group['properties'].items():
                        param_info = [
                            f"  - {param_name}:",
                            f"    Type: {param_def.get('type', 'string')}",
                            f"    Description: {param_def.get('description', 'No description')}",
                            f"    Required: {param_name in param_group.get('required', [])}",
                        ]
                        if 'default' in param_def:
                            param_info.append(f"    Default: {param_def['default']}")
                        if 'enum' in param_def:
                            param_info.append(f"    Options: {', '.join(map(str, param_def['enum']))}")
                        if 'pattern' in param_def:
                            param_info.append(f"    Pattern: {param_def['pattern']}")
                        
                        doc_parts.extend(param_info)
        
        # Add steps information
        doc_parts.extend(["", "Template Steps:"])
        for i, step in enumerate(template.steps, 1):
            doc_parts.extend([
                f"  {i}. {step.get('name', f'Step {i}')}",
                f"     Action: {step.get('action', 'unknown')}",
                f"     Description: {step.get('input', {})}"
            ])
        
        return "\n".join(doc_parts)
    
    async def get_template_context(self, template_type: str, user_query: str) -> Dict[str, Any]:
        """Get relevant template context using RAG"""
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        # Search for relevant template information
        search_query = f"{template_type} {user_query}"
        relevant_docs = self.vector_store.similarity_search(
            search_query, 
            k=3,
            filter={"template_type": template_type} if template_type != "unknown" else None
        )
        
        # Get the full template if we have an exact match
        matching_template = None
        for template_name, template in self.templates.items():
            if template_type.lower() in template_name.lower() or \
               template_type.lower() in template.yaml_content.get("metadata", {}).get("tags", []):
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
                "required_fields": self._extract_required_fields(matching_template),
                "optional_fields": self._extract_optional_fields(matching_template),
                "validation_rules": self._extract_validation_rules(matching_template)
            }
        
        return context
    
    def _extract_required_fields(self, template: BackstageTemplate) -> List[str]:
        """Extract required field names from template"""
        required_fields = []
        parameters = template.parameters
        
        if isinstance(parameters, list):
            for param_group in parameters:
                required_fields.extend(param_group.get('required', []))
        
        return required_fields
    
    def _extract_optional_fields(self, template: BackstageTemplate) -> List[str]:
        """Extract optional field names from template"""
        optional_fields = []
        required_fields = set(self._extract_required_fields(template))
        parameters = template.parameters
        
        if isinstance(parameters, list):
            for param_group in parameters:
                if 'properties' in param_group:
                    for param_name in param_group['properties'].keys():
                        if param_name not in required_fields:
                            optional_fields.append(param_name)
        
        return optional_fields
    
    def _extract_validation_rules(self, template: BackstageTemplate) -> Dict[str, Any]:
        """Extract validation rules from template"""
        validation_rules = {}
        parameters = template.parameters
        
        if isinstance(parameters, list):
            for param_group in parameters:
                if 'properties' in param_group:
                    for param_name, param_def in param_group['properties'].items():
                        rules = {}
                        
                        if 'pattern' in param_def:
                            rules['pattern'] = param_def['pattern']
                        if 'minimum' in param_def:
                            rules['minimum'] = param_def['minimum']
                        if 'maximum' in param_def:
                            rules['maximum'] = param_def['maximum']
                        if 'enum' in param_def:
                            rules['allowed_values'] = param_def['enum']
                        if 'type' in param_def:
                            rules['type'] = param_def['type']
                        
                        if rules:
                            validation_rules[param_name] = rules
        
        return validation_rules


# ====================================================================
# DYNAMIC MCP DISCOVERY AND TOOL CALLING
# ====================================================================

@dataclass
class MCPServerInfo:
    """Information about an MCP server"""
    name: str
    url: str
    capabilities: List[str]
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class DynamicMCPDiscovery:
    """Handles dynamic discovery and interaction with MCP servers"""
    
    def __init__(self):
        self.client_factory = ClientFactory()
        self.discovered_servers: Dict[str, MCPServerInfo] = {}
        self.active_clients: Dict[str, MCPClient] = {}
        
    async def discover_mcp_servers(self) -> Dict[str, MCPServerInfo]:
        """Dynamically discover available MCP servers"""
        
        # In real implementation, this might:
        # 1. Query service discovery (Consul, etcd, Kubernetes)
        # 2. Read from configuration files
        # 3. Use environment variables
        # 4. Call registry service
        
        # For now, we'll use environment-based discovery
        server_configs = self._load_server_configs()
        
        discovered = {}
        for server_name, config in server_configs.items():
            try:
                # Try to connect and get server info
                server_info = await self._probe_mcp_server(server_name, config)
                if server_info:
                    discovered[server_name] = server_info
                    logger.info(f"Discovered MCP server: {server_name}")
            except Exception as e:
                logger.warning(f"Failed to discover server {server_name}: {e}")
        
        self.discovered_servers = discovered
        return discovered
    
    def _load_server_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load MCP server configurations from various sources"""
        
        # Environment-based configuration
        servers = {}
        
        # Check for individual server URLs
        if os.getenv("TEMPLATE_AGENT_URL"):
            servers["TemplateAgent"] = {
                "url": os.getenv("TEMPLATE_AGENT_URL"),
                "type": "template_management"
            }
        
        if os.getenv("CATALOG_AGENT_URL"):
            servers["CatalogAgent"] = {
                "url": os.getenv("CATALOG_AGENT_URL"),
                "type": "catalog_management"
            }
        
        if os.getenv("ACTION_AGENT_URL"):
            servers["ActionAgent"] = {
                "url": os.getenv("ACTION_AGENT_URL"),
                "type": "action_execution"
            }
        
        # Check for JSON configuration
        servers_json = os.getenv("MCP_SERVERS_CONFIG")
        if servers_json:
            try:
                additional_servers = json.loads(servers_json)
                servers.update(additional_servers)
            except json.JSONDecodeError:
                logger.warning("Invalid MCP_SERVERS_CONFIG JSON")
        
        # Default local servers for development
        if not servers:
            servers = {
                "TemplateAgent": {
                    "url": "http://localhost:8001",
                    "type": "template_management"
                },
                "CatalogAgent": {
                    "url": "http://localhost:8002", 
                    "type": "catalog_management"
                },
                "ActionAgent": {
                    "url": "http://localhost:8003",
                    "type": "action_execution"
                }
            }
        
        return servers
    
    async def _probe_mcp_server(self, server_name: str, config: Dict[str, Any]) -> Optional[MCPServerInfo]:
        """Probe an MCP server to get its capabilities"""
        try:
            # Create client connection
            client = await self.client_factory.create(server_name, config["url"])
            
            # Get server capabilities
            capabilities = await self._get_server_capabilities(client, server_name)
            
            # Store active client
            self.active_clients[server_name] = client
            
            return MCPServerInfo(
                name=server_name,
                url=config["url"],
                capabilities=capabilities,
                status="healthy",
                metadata={
                    "type": config.get("type", "unknown"),
                    "probed_at": asyncio.get_event_loop().time()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to probe {server_name}: {e}")
            return None
    
    async def _get_server_capabilities(self, client: MCPClient, server_name: str) -> List[str]:
        """Get available tools/capabilities from an MCP server"""
        try:
            # Try to call a standard capabilities endpoint
            result = await client.call_tool("list_capabilities", {})
            if result and "capabilities" in result:
                return result["capabilities"]
        except:
            pass
        
        # Fallback: try to introspect available tools
        try:
            # This would depend on the MCP client's introspection capabilities
            tools = await client.list_tools()
            return [tool.name for tool in tools] if tools else []
        except:
            pass
        
        # Default capabilities based on server name
        default_capabilities = {
            "TemplateAgent": [
                "get_template_params",
                "validate_template_params",
                "list_available_templates"
            ],
            "CatalogAgent": [
                "get_developer_context",
                "get_team_info", 
                "get_organization_standards"
            ],
            "ActionAgent": [
                "create_backstage_component",
                "validate_component_creation",
                "get_component_status"
            ]
        }
        
        return default_capabilities.get(server_name, ["unknown"])
    
    async def call_mcp_tool(self, server_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically call a tool on an MCP server"""
        if server_name not in self.active_clients:
            # Try to reconnect
            if server_name in self.discovered_servers:
                server_info = self.discovered_servers[server_name]
                client = await self.client_factory.create(server_name, server_info.url)
                self.active_clients[server_name] = client
            else:
                raise ValueError(f"MCP server {server_name} not discovered")
        
        client = self.active_clients[server_name]
        
        try:
            result = await client.call_tool(tool_name, parameters)
            logger.info(f"Called {server_name}.{tool_name} successfully")
            return result
        except Exception as e:
            logger.error(f"Error calling {server_name}.{tool_name}: {e}")
            raise
    
    def get_servers_by_capability(self, capability: str) -> List[str]:
        """Find servers that have a specific capability"""
        matching_servers = []
        for server_name, server_info in self.discovered_servers.items():
            if capability in server_info.capabilities:
                matching_servers.append(server_name)
        return matching_servers
    
    def get_server_capabilities(self, server_name: str) -> List[str]:
        """Get capabilities of a specific server"""
        if server_name in self.discovered_servers:
            return self.discovered_servers[server_name].capabilities
        return []


# ====================================================================
# ENHANCED ORCHESTRATOR WITH RAG AND DYNAMIC MCP
# ====================================================================

class EnhancedOrchestrationExecutor:
    """Enhanced executor with RAG template context and dynamic MCP discovery"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=0.1
        )
        
        # RAG and MCP components
        self.template_rag = TemplateRAGManager()
        self.mcp_discovery = DynamicMCPDiscovery()
        
        # Session management
        self.sessions = {}
        
    async def initialize(self):
        """Initialize RAG and MCP discovery systems"""
        logger.info("Initializing Enhanced Orchestrator...")
        
        # Initialize template RAG
        await self.template_rag.initialize()
        logger.info(f"Loaded {len(self.template_rag.templates)} templates")
        
        # Discover MCP servers
        discovered = await self.mcp_discovery.discover_mcp_servers()
        logger.info(f"Discovered {len(discovered)} MCP servers")
        
        logger.info("Enhanced Orchestrator initialized successfully")
    
    async def process_user_request(self, user_input: str, session_id: str = "default") -> str:
        """Process user request with RAG context and dynamic MCP calls"""
        
        # Get or create session state
        if session_id not in self.sessions:
            from models.workflow_state import WorkflowState, ConversationState
            self.sessions[session_id] = WorkflowState()
        
        state = self.sessions[session_id]
        
        if state.current_state.value == "initial":
            return await self._handle_initial_request_with_rag(user_input, state)
        elif state.current_state.value == "awaiting_confirmation":
            return await self._handle_user_confirmation_with_mcp(user_input, state)
        else:
            return "I'm currently processing your request. Please wait..."
    
    async def _handle_initial_request_with_rag(self, user_input: str, state) -> str:
        """Handle initial request using RAG context for template analysis"""
        
        # Step 1: Use LLM to analyze request and determine template type
        analysis_prompt = f"""
        Analyze this user request to determine what type of Backstage template they want to create.
        
        User request: "{user_input}"
        
        Available template types from our catalog:
        {list(self.template_rag.templates.keys())}
        
        Respond with JSON:
        {{
            "template_type": "exact_template_name_from_catalog",
            "confidence": 0.0-1.0,
            "reasoning": "why this template was selected",
            "inferred_parameters": {{"param": "value", ...}}
        }}
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert at analyzing software development requests and mapping them to Backstage templates."),
            HumanMessage(content=analysis_prompt)
        ])
        
        try:
            analysis = json.loads(response.content)
            template_type = analysis.get("template_type", "lambda-template")
        except json.JSONDecodeError:
            template_type = "lambda-template"  # Default fallback
        
        # Step 2: Get RAG context for the template
        template_context = await self.template_rag.get_template_context(template_type, user_input)
        
        # Update state
        state.user_request = user_input
        state.template_type = template_type
        state.current_state = "gathering_context"  # Would need to import enum
        
        # Step 3: Dynamically call MCP servers based on discovered capabilities
        return await self._gather_context_with_dynamic_mcp(state, template_context)
    
    async def _gather_context_with_dynamic_mcp(self, state, template_context: Dict[str, Any]) -> str:
        """Gather context using dynamically discovered MCP servers"""
        
        gathered_context = {"template_rag_context": template_context}
        
        # Find servers that can provide template context
        template_servers = self.mcp_discovery.get_servers_by_capability("get_template_params")
        catalog_servers = self.mcp_discovery.get_servers_by_capability("get_developer_context")
        
        # Parallel MCP calls
        tasks = []
        
        # Call template servers
        for server in template_servers:
            tasks.append(
                self._safe_mcp_call(
                    server, 
                    "get_template_params", 
                    {"template_type": state.template_type}
                )
            )
        
        # Call catalog servers
        for server in catalog_servers:
            tasks.append(
                self._safe_mcp_call(
                    server,
                    "get_developer_context", 
                    {"user_id": os.getenv("USER_ID", "current_user")}
                )
            )
        
        # Execute all MCP calls in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                server_name = template_servers[i] if i < len(template_servers) else catalog_servers[i - len(template_servers)]
                if isinstance(result, Exception):
                    logger.warning(f"MCP call to {server_name} failed: {result}")
                    gathered_context[f"{server_name}_error"] = str(result)
                else:
                    gathered_context[f"{server_name}_data"] = result
        
        # Step 4: Use LLM with RAG context to intelligently map fields
        return await self._process_context_with_llm_rag(state, gathered_context)
    
    async def _safe_mcp_call(self, server_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely call MCP tool with error handling"""
        try:
            return await self.mcp_discovery.call_mcp_tool(server_name, tool_name, parameters)
        except Exception as e:
            logger.error(f"MCP call failed {server_name}.{tool_name}: {e}")
            raise e
    
    async def _process_context_with_llm_rag(self, state, gathered_context: Dict[str, Any]) -> str:
        """Process gathered context using LLM with RAG template information"""
        
        template_rag_context = gathered_context.get("template_rag_context", {})
        template_metadata = template_rag_context.get("template_metadata", {})
        
        # Build comprehensive context for LLM
        context_prompt = f"""
        You are helping create a Backstage component using the template: {state.template_type}
        
        User Request: {state.user_request}
        
        TEMPLATE INFORMATION FROM RAG:
        {json.dumps(template_metadata, indent=2)}
        
        RELEVANT TEMPLATE CONTENT:
        {chr(10).join(template_rag_context.get("relevant_content", []))}
        
        GATHERED MCP CONTEXT:
        {json.dumps({k:v for k,v in gathered_context.items() if k != "template_rag_context"}, indent=2)}
        
        Based on the template requirements and available context:
        1. Identify which fields can be pre-filled from the available context
        2. Determine which fields still need user input
        3. Apply validation rules from the template
        4. Provide intelligent defaults where possible
        
        Respond with JSON:
        {{
            "pre_filled_fields": {{"field_name": "suggested_value", ...}},
            "missing_fields": ["field1", "field2", ...],
            "field_explanations": {{"field": "why this value was chosen or why it's needed", ...}},
            "validation_warnings": ["any validation issues", ...],
            "confirmation_message": "Human readable summary for user confirmation"
        }}
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert at mapping context to Backstage template parameters using template schemas and organizational context."),
            HumanMessage(content=context_prompt)
        ])
        
        try:
            mapping = json.loads(response.content)
            
            # Update state with processed information
            state.required_fields = mapping.get("pre_filled_fields", {})
            state.missing_fields = mapping.get("missing_fields", [])
            state.gathered_context = gathered_context
            state.current_state = "awaiting_confirmation"  # Would need proper enum import
            
            # Build user-friendly response
            confirmation_msg = mapping.get("confirmation_message", "I've analyzed your request and gathered context.")
            
            # Format pre-filled fields
            prefilled_text = ""
            if mapping.get("pre_filled_fields"):
                prefilled_text = "\n\n📋 **Pre-filled from context:**\n"
                for field, value in mapping["pre_filled_fields"].items():
                    explanation = mapping.get("field_explanations", {}).get(field, "")
                    prefilled_text += f"• **{field}**: `{value}`"
                    if explanation:
                        prefilled_text += f" _{explanation}_"
                    prefilled_text += "\n"
            
            # Format missing fields
            missing_text = ""
            if mapping.get("missing_fields"):
                missing_text = "\n\n❓ **Still needed:**\n"
                for field in mapping["missing_fields"]:
                    explanation = mapping.get("field_explanations", {}).get(field, "")
                    missing_text += f"• **{field}**"
                    if explanation:
                        missing_text += f": {explanation}"
                    missing_text += "\n"
            
            # Format validation warnings
            warnings_text = ""
            if mapping.get("validation_warnings"):
                warnings_text = "\n\n⚠️ **Validation Notes:**\n"
                for warning in mapping["validation_warnings"]:
                    warnings_text += f"• {warning}\n"
            
            return f"""{confirmation_msg}
            
🎯 **Template**: {state.template_type}
{prefilled_text}{missing_text}{warnings_text}
            
Please review and confirm, or provide the missing information.
Type **'confirm'** to proceed, or provide corrections/missing details."""
            
        except json.JSONDecodeError as e:
            logger.error(f"LLM response parsing failed: {e}")
            return f"I've gathered context for your {state.template_type} template, but need to ask you for some details. What specific information would you like to provide?"
    
    async def _handle_user_confirmation_with_mcp(self, user_input: str, state) -> str:
        """Handle user confirmation and create component via MCP"""
        
        if user_input.lower().strip() == "confirm":
            # User confirmed, proceed to creation
            return await self._create_component_with_action_agent(state)
        else:
            # Parse user updates using LLM with template context
            return await self._update_fields_with_llm_validation(user_input, state)
    
    async def _update_fields_with_llm_validation(self, user_input: str, state) -> str:
        """Update fields using LLM with template validation"""
        
        template_context = state.gathered_context.get("template_rag_context", {})
        validation_rules = template_context.get("template_metadata", {}).get("validation_rules", {})
        
        update_prompt = f"""
        The user provided this input to update template fields: "{user_input}"
        
        Current fields: {json.dumps(state.required_fields, indent=2)}
        Missing fields: {state.missing_fields}
        
        Template validation rules: {json.dumps(validation_rules, indent=2)}
        
        Parse the user input and update fields according to template validation rules.
        
        Respond with JSON:
        {{
            "updated_fields": {{"field_name": "new_value", ...}},
            "still_missing": ["field1", "field2", ...],
            "validation_errors": ["field: error message", ...],
            "ready_to_proceed": true/false,
            "explanation": "what was updated and any issues"
        }}
        """
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert at parsing user input and validating it against Backstage template requirements."),
            HumanMessage(content=update_prompt)
        ])
        
        try:
            update = json.loads(response.content)
            
            # Apply updates
            for field, value in update.get("updated_fields", {}).items():
                state.required_fields[field] = value
            
            state.missing_fields = update.get("still_missing", [])
            
            # Check if ready to proceed
            if update.get("ready_to_proceed", False) and not state.missing_fields and not update.get("validation_errors"):
                return await self._create_component_with_action_agent(state)
            
            # Build response about what was updated
            response_parts = [update.get("explanation", "I've processed your input.")]
            
            if update.get("validation_errors"):
                response_parts.append("\n❌ **Validation Errors:**")
                for error in update["validation_errors"]:
                    response_parts.append(f"• {error}")
            
            if state.missing_fields:
                response_parts.append(f"\n❓ **Still needed:** {', '.join(state.missing_fields)}")
            
            response_parts.append("\nPlease provide the missing/corrected information, or type 'confirm' to proceed.")
            
            return "\n".join(response_parts)
            
        except json.JSONDecodeError:
            return "I couldn't parse your input. Please try again with the missing information, or type 'confirm' to proceed with current values."
    
    async def _create_component_with_action_agent(self, state) -> str:
        """Create component using ActionAgent MCP server"""
        
        # Find action servers
        action_servers = self.mcp_discovery.get_servers_by_capability("create_backstage_component")
        
        if not action_servers:
            # Fallback creation simulation
            return await self._simulate_component_creation(state)
        
        # Use the first available action server
        action_server = action_servers[0]
        
        try:
            creation_result = await self.mcp_discovery.call_mcp_tool(
                action_server,
                "create_backstage_component",
                {
                    "template_type": state.template_type,
                    "parameters": state.required_fields
                }
            )
            
            state.current_state = "completed"
            
            component_name = state.required_fields.get("component_name", "your-component")
            
            # Format success response
            response_parts = [
                f"✅ **Success!** Your {state.template_type} component '**{component_name}**' has been created!",
                "",
                "📋 **Details:**"
            ]
            
            if creation_result.get("catalog_url"):
                response_parts.append(f"• **Catalog**: {creation_result['catalog_url']}")
            if creation_result.get("repository_url"):
                response_parts.append(f"• **Repository**: {creation_result['repository_url']}")
            
            if creation_result.get("created_files"):
                response_parts.append(f"• **Files Created**: {', '.join(creation_result['created_files'])}")
            
            if creation_result.get("next_steps"):
                response_parts.append("\n🚀 **Next Steps:**")
                for step in creation_result["next_steps"]:
                    response_parts.append(f"• {step}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Component creation failed: {e}")
            return await self._simulate_component_creation(state)
    
    async def _simulate_component_creation(self, state) -> str:
        """Simulate component creation when ActionAgent is not available"""
        component_name = state.required_fields.get("component_name", "your-component")
        
        return f"""✅ **Component Ready for Creation!**

**Template**: {state.template_type}
**Component**: {component_name}

**Parameters:**
{json.dumps(state.required_fields, indent=2)}

*Note: This is a simulation. In production, this would create the actual Backstage component via the ActionAgent MCP server.*

**What would happen next:**
• Generate component files from template
• Create GitHub repository 
• Register in Backstage catalog
• Set up CI/CD pipeline
• Apply security policies"""


# ====================================================================
# SAMPLE MCP SERVERS WITH REAL TEMPLATE INTEGRATION
# ====================================================================

# Enhanced TemplateAgent with RAG integration
enhanced_template_app = FastMCP("EnhancedTemplateAgent")
template_rag_manager = TemplateRAGManager()

@enhanced_template_app.startup()
async def startup():
    """Initialize template RAG on startup"""
    await template_rag_manager.initialize()

@enhanced_template_app.tool()
async def get_template_params(template_type: str) -> Dict[str, Any]:
    """Get template parameters using RAG context"""
    context = await template_rag_manager.get_template_context(template_type, "")
    return context.get("template_metadata", {})

@enhanced_template_app.tool()
async def search_templates(query: str) -> Dict[str, Any]:
    """Search templates using RAG similarity search"""
    if not template_rag_manager.vector_store:
        return {"error": "Vector store not initialized"}
    
    docs = template_rag_manager.vector_store.similarity_search(query, k=5)
    
    return {
        "query": query,
        "results": [
            {
                "template_name": doc.metadata.get("template_name"),
                "relevance_score": 1.0,  # FAISS doesn't return scores by default
                "content_preview": doc.page_content[:200],
                "tags": doc.metadata.get("tags", [])
            }
            for doc in docs
        ]
    }

@enhanced_template_app.tool()
async def validate_template_params(template_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate parameters against template schema"""
    context = await template_rag_manager.get_template_context(template_type, "")
    validation_rules = context.get("template_metadata", {}).get("validation_rules", {})
    
    errors = []
    warnings = []
    
    for param_name, param_value in params.items():
        if param_name in validation_rules:
            rules = validation_rules[param_name]
            
            # Type validation
            if "type" in rules:
                expected_type = rules["type"]
                if expected_type == "integer" and not isinstance(param_value, int):
                    errors.append(f"{param_name}: Expected integer, got {type(param_value).__name__}")
                elif expected_type == "string" and not isinstance(param_value, str):
                    errors.append(f"{param_name}: Expected string, got {type(param_value).__name__}")
            
            # Pattern validation
            if "pattern" in rules and isinstance(param_value, str):
                import re
                if not re.match(rules["pattern"], param_value):
                    errors.append(f"{param_name}: Does not match required pattern {rules['pattern']}")
            
            # Range validation
            if "minimum" in rules and isinstance(param_value, (int, float)):
                if param_value < rules["minimum"]:
                    errors.append(f"{param_name}: Must be at least {rules['minimum']}")
            
            if "maximum" in rules and isinstance(param_value, (int, float)):
                if param_value > rules["maximum"]:
                    errors.append(f"{param_name}: Must not exceed {rules['maximum']}")
            
            # Enum validation
            if "allowed_values" in rules:
                if param_value not in rules["allowed_values"]:
                    errors.append(f"{param_name}: Must be one of {rules['allowed_values']}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "validated_params": params
    }

@enhanced_template_app.tool()
async def list_capabilities() -> Dict[str, Any]:
    """List server capabilities for dynamic discovery"""
    return {
        "capabilities": [
            "get_template_params",
            "search_templates", 
            "validate_template_params",
            "list_capabilities"
        ],
        "server_type": "template_management",
        "version": "1.0.0",
        "supports_rag": True,
        "template_count": len(template_rag_manager.templates)
    }


# ====================================================================
# MAIN APPLICATION WITH ENHANCED FEATURES
# ====================================================================

async def main():
    """Main function with enhanced RAG and dynamic MCP capabilities"""
    import sys
    
    # Initialize enhanced orchestrator
    orchestrator = EnhancedOrchestrationExecutor()
    await orchestrator.initialize()
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # CLI Mode
        print("🚀 Enhanced Backstage Template Orchestrator CLI")
        print("=" * 60)
        print("✨ Features: RAG Template Context + Dynamic MCP Discovery")
        print(f"📚 Loaded Templates: {list(orchestrator.template_rag.templates.keys())}")
        print(f"🔌 Discovered MCP Servers: {list(orchestrator.mcp_discovery.discovered_servers.keys())}")
        print()
        print("Type 'exit' to quit, 'reset' to start over, 'help' for commands")
        print()
        
        session_id = "cli_session"
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'reset':
                    if session_id in orchestrator.sessions:
                        del orchestrator.sessions[session_id]
                    print("🔄 Session reset!")
                    continue
                elif user_input.lower() == 'help':
                    print("""
Available commands:
• exit - Quit the CLI
• reset - Reset conversation state  
• help - Show this help message
• templates - List available templates
• servers - List discovered MCP servers

Example requests:
• "I want to create a lambda function for processing user events"
• "Create a microservice for user authentication with a PostgreSQL database"
• "I need a new API gateway component"
                    """)
                    continue
                elif user_input.lower() == 'templates':
                    templates = list(orchestrator.template_rag.templates.keys())
                    print(f"📚 Available templates: {', '.join(templates)}")
                    continue
                elif user_input.lower() == 'servers':
                    servers = orchestrator.mcp_discovery.discovered_servers
                    for name, info in servers.items():
                        print(f"🔌 {name}: {info.status} - {', '.join(info.capabilities)}")
                    continue
                elif not user_input:
                    continue
                
                print("🤖 Agent: Processing with RAG context...")
                response = await orchestrator.process_user_request(user_input, session_id)
                print(f"🤖 Agent: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
    else:
        # Server mode
        print("🚀 Starting Enhanced Backstage Template Orchestrator Server...")
        print("✨ Features: RAG Template Context + Dynamic MCP Discovery") 
        print(f"📚 Templates: {list(orchestrator.template_rag.templates.keys())}")
        print(f"🔌 MCP Servers: {list(orchestrator.mcp_discovery.discovered_servers.keys())}")
        print("Use 'python script.py cli' to run in CLI mode")
        
        # Create FastAPI app with orchestrator
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        
        app = FastAPI(
            title="Enhanced Backstage Template Orchestrator",
            description="RAG-powered orchestration with dynamic MCP discovery",
            version="2.0.0"
        )
        
        class ChatRequest(BaseModel):
            message: str
            session_id: str = "default"
        
        class ChatResponse(BaseModel):
            response: str
            session_id: str
            template_type: Optional[str] = None
            current_state: Optional[str] = None
        
        @app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest):
            try:
                response = await orchestrator.process_user_request(
                    request.message, 
                    request.session_id
                )
                
                session_state = orchestrator.sessions.get(request.session_id)
                return ChatResponse(
                    response=response,
                    session_id=request.session_id,
                    template_type=session_state.template_type if session_state else None,
                    current_state=session_state.current_state if session_state else None
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/templates")
        async def list_templates():
            """List available templates"""
            return {
                "templates": list(orchestrator.template_rag.templates.keys()),
                "count": len(orchestrator.template_rag.templates)
            }
        
        @app.get("/servers")
        async def list_servers():
            """List discovered MCP servers"""
            return orchestrator.mcp_discovery.discovered_servers
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "service": "enhanced-backstage-orchestrator",
                "version": "2.0.0",
                "features": ["rag", "dynamic_mcp_discovery"],
                "templates_loaded": len(orchestrator.template_rag.templates),
                "mcp_servers_discovered": len(orchestrator.mcp_discovery.discovered_servers)
            }
        
        import uvicorn
        uvicorn.run(
            app,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 8000)),
            log_level="info"
        )

if __name__ == "__main__":
    asyncio.run(main())