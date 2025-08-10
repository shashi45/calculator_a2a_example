# Agentic Orchestration Project Structure

This README explains how to organize the codebase based on the sequence diagram components, splitting the monolithic code into proper modules and files.

## 📁 Project Structure

```
backstage-orchestrator/
├── README.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── main.py                           # Application entry point
├── cli.py                           # CLI testing interface
├── agents/
│   ├── __init__.py
│   ├── conversational_agent.py     # ConversationalAgent
│   ├── orchestrator.py             # Orchestrator (Host Agent)
│   └── base_agent.py               # Base agent functionality
├── mcp_servers/
│   ├── __init__.py
│   ├── template_agent.py           # TemplateAgent MCP Server
│   ├── catalog_agent.py            # CatalogAgent MCP Server
│   ├── action_agent.py             # ActionAgent MCP Server
│   └── registry.py                 # MCP Server Registry
├── models/
│   ├── __init__.py
│   ├── workflow_state.py           # WorkflowState and enums
│   └── agent_cards.py              # Agent Card definitions
├── utils/
│   ├── __init__.py
│   ├── llm_client.py               # LLM integration utilities
│   └── config.py                   # Configuration management
└── tests/
    ├── __init__.py
    ├── test_orchestrator.py
    ├── test_mcp_servers.py
    └── integration/
        └── test_full_flow.py
```

## 📄 File Contents Breakdown

### 1. `main.py` - Application Entry Point
**Components:** A2AStarterApplication startup, FastAPI integration
```python
#!/usr/bin/env python3
"""
Main application entry point for the Agentic Orchestration service
Starts the A2AStarterApplication with uvicorn
"""

import os
import uvicorn
from fastapi import FastAPI
from fastmcp.applications import A2AStarterApplication
from agents.orchestrator import OrchestrationHostAgent
from utils.config import load_config

def create_a2a_application() -> A2AStarterApplication:
    """Create and configure the A2A Starter Application"""
    # FastAPI setup
    # MCP server mounting
    # A2A application creation
    pass

async def main():
    """Main function - server mode only"""
    config = load_config()
    a2a_app = create_a2a_application()
    
    uvicorn.run(
        a2a_app.app,
        host=config.host,
        port=config.port,
        log_level="info"
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 2. `cli.py` - CLI Testing Interface
**Components:** CLI for testing multi-turn conversations
```python
#!/usr/bin/env python3
"""
CLI interface for testing the multi-turn conversation flow
"""

import asyncio
from agents.conversational_agent import ConversationalAgent
from agents.orchestrator import OrchestrationHostAgent

class CLITestInterface:
    """CLI interface for testing the multi-turn conversation flow"""
    
    def __init__(self):
        self.conversational_agent = ConversationalAgent()
        self.orchestrator = OrchestrationHostAgent()
    
    async def run_cli(self):
        """Run the interactive CLI following sequence diagram flow"""
        # User → ConversationalAgent → Orchestrator flow
        pass

async def main():
    cli = CLITestInterface()
    await cli.run_cli()

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. `agents/conversational_agent.py` - ConversationalAgent
**Components:** User-facing interface, forwards requests to Orchestrator
```python
"""
ConversationalAgent - User-facing interface component
Handles user input and forwards to Orchestrator
"""

from typing import Dict, Any
from agents.orchestrator import OrchestrationHostAgent

class ConversationalAgent:
    """
    ConversationalAgent from sequence diagram
    - Receives user input
    - Forwards to Orchestrator
    - Returns responses to user
    """
    
    def __init__(self):
        self.orchestrator = OrchestrationHostAgent()
        self.session_state = {}
    
    async def process_user_message(self, message: str, session_id: str = "default") -> str:
        """
        Step 1: User → ConversationalAgent: User provides initial prompt
        Step 2: ConversationalAgent → Orchestrator: Forward user request
        """
        # Forward to orchestrator
        response = await self.orchestrator.handle_message(message, session_id)
        
        # Step 10: ConversationalAgent → User: Return final response
        return response
    
    async def prompt_user_for_field(self, field_name: str, context: Dict[str, Any]) -> str:
        """
        Step 7: Orchestrator → ConversationalAgent: Ask next missing field
        Step 8: ConversationalAgent → User: Prompt for field value
        """
        # In CLI mode, this prompts directly
        # In API mode, this returns a structured response for the frontend
        pass
```

### 4. `agents/orchestrator.py` - Orchestrator (Host Agent)
**Components:** Core orchestration logic, LLM reasoning, MCP coordination
```python
"""
Orchestrator - Central coordination component with LLM reasoning
Implements the Host Agent with Agent Card + Executor pattern
"""

import json
from typing import Dict, Any, List
from fastmcp.client import ClientFactory
from langchain_openai import ChatOpenAI
from models.workflow_state import WorkflowState, ConversationState
from models.agent_cards import OrchestrationAgentCard
from utils.llm_client import LLMClient

class OrchestrationExecutor:
    """
    Executor class managing orchestration logic per sequence diagram
    """
    
    def __init__(self):
        self.llm = LLMClient()
        self.client_factory = ClientFactory()
        self.sessions = {}  # session_id -> WorkflowState
    
    async def process_user_request(self, user_input: str, session_id: str) -> str:
        """
        Main orchestration following sequence diagram:
        - Step 3: Orchestrator → LLM reasoning - build LangGraph plan
        - Step 4: Orchestrator → MCPRegistry: Discover available servers
        - Steps 5-6: Parallel MCP server calls
        - Step 7: Map fields, detect missing/ambiguous
        - Steps 8-9: Multi-turn conversation loop
        """
        state = self.sessions.get(session_id, WorkflowState())
        self.sessions[session_id] = state
        
        if state.current_state == ConversationState.INITIAL:
            return await self._handle_initial_request(user_input, state)
        # ... other state handlers
    
    async def _fetch_context_parallel(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Steps 5-6: Parallel context gathering
        - Orchestrator → TemplateAgent_MCP: A2A call via MCP
        - Orchestrator → CatalogAgent_MCP: A2A call via MCP
        """
        template_client = await self.client_factory.create("TemplateAgent")
        catalog_client = await self.client_factory.create("CatalogAgent")
        
        template_task = self._fetch_template_context(template_client, state.template_type)
        catalog_task = self._fetch_catalog_context(catalog_client)
        
        template_context, catalog_context = await asyncio.gather(
            template_task, catalog_task, return_exceptions=True
        )
        
        return {
            "template_info": template_context,
            "catalog_info": catalog_context
        }

class OrchestrationHostAgent:
    """
    Host Agent implementing Agent Card pattern
    """
    
    def __init__(self):
        self.executor = OrchestrationExecutor()
        self.agent_card = OrchestrationAgentCard()
    
    async def handle_message(self, message: str, session_id: str) -> str:
        """Route messages to executor"""
        return await self.executor.process_user_request(message, session_id)
```

### 5. `mcp_servers/template_agent.py` - TemplateAgent MCP Server
**Components:** Template parameter management
```python
"""
TemplateAgent MCP Server
Manages template schemas and parameters
"""

from fastmcp import FastMCP
from typing import Dict, Any, List

app = FastMCP("TemplateAgent")

@app.tool()
async def get_template_params(template_type: str) -> Dict[str, Any]:
    """
    Fetch template.yaml parameters for specified template type
    Used in: Orchestrator → TemplateAgent_MCP: A2A call via MCP
    """
    template_schemas = {
        "lambda": {
            "required_params": ["component_name", "description", "owner_team", "runtime"],
            "optional_params": ["memory_size", "timeout", "environment_variables"],
            "validation_rules": {
                "component_name": r"^[a-z][a-z0-9-]*[a-z0-9]$",
                "runtime": ["python3.9", "python3.11", "nodejs18.x", "nodejs20.x"]
            }
        },
        "microservice": {
            "required_params": ["service_name", "description", "owner_team", "port"],
            "optional_params": ["health_check_path", "cpu_limits", "memory_limits"],
            "validation_rules": {
                "port": {"min": 3000, "max": 9999}
            }
        }
    }
    
    return template_schemas.get(template_type, {})

@app.tool() 
async def validate_template_params(template_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate template parameters against schema"""
    # Validation logic
    pass

@app.tool()
async def list_available_templates() -> List[str]:
    """List all available template types"""
    return ["lambda", "microservice", "database", "frontend", "api-gateway"]
```

### 6. `mcp_servers/catalog_agent.py` - CatalogAgent MCP Server
**Components:** Organizational context provider
```python
"""
CatalogAgent MCP Server  
Provides organizational context (teams, standards, etc.)
"""

from fastmcp import FastMCP
from typing import Dict, Any, List

app = FastMCP("CatalogAgent")

@app.tool()
async def get_developer_context(user_id: str = "current_user") -> Dict[str, Any]:
    """
    Fetch developer/team context from Backstage catalog
    Used in: Orchestrator → CatalogAgent_MCP: A2A call via MCP
    """
    # In real implementation, this would query Backstage API
    mock_context = {
        "user_info": {
            "id": user_id,
            "name": "John Developer",
            "email": "john@company.com",
            "teams": ["platform-team", "backend-team"]
        },
        "organization": {
            "default_namespace": "my-company",
            "naming_conventions": {
                "components": "kebab-case",
                "apis": "camelCase"
            },
            "required_tags": ["environment", "team", "lifecycle"],
            "available_environments": ["development", "staging", "production"]
        },
        "team_defaults": {
            "platform-team": {
                "default_owner": "platform-team",
                "default_lifecycle": "production",
                "preferred_runtime": "python3.11"
            }
        }
    }
    
    return mock_context

@app.tool()
async def get_team_info(team_name: str) -> Dict[str, Any]:
    """Get specific team information"""
    pass

@app.tool()
async def get_organization_standards() -> Dict[str, Any]:
    """Get organization-wide standards and conventions"""
    pass
```

### 7. `mcp_servers/action_agent.py` - ActionAgent MCP Server
**Components:** Backstage component creation
```python
"""
ActionAgent MCP Server
Executes Backstage component creation
"""

from fastmcp import FastMCP
from typing import Dict, Any

app = FastMCP("ActionAgent")

@app.tool()
async def create_backstage_component(template_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create Backstage component with all parameters
    Used in: Orchestrator → ActionAgent_MCP: A2A call via MCP
    Final step in sequence diagram
    """
    component_name = parameters.get("component_name", "unnamed-component")
    
    # In real implementation, this would:
    # 1. Generate component files from template
    # 2. Create repository (if needed)
    # 3. Register in Backstage catalog
    # 4. Set up CI/CD pipelines
    # 5. Apply security policies
    
    creation_result = {
        "status": "success",
        "component_id": f"{parameters.get('owner_team', 'default')}/{component_name}",
        "catalog_url": f"https://backstage.company.com/catalog/default/component/{component_name}",
        "repository_url": f"https://github.com/company/{component_name}",
        "created_files": [
            "catalog-info.yaml",
            "README.md", 
            f"src/{template_type}_template/",
            ".github/workflows/ci.yml"
        ],
        "next_steps": [
            "Review generated code",
            "Customize configuration", 
            "Run initial deployment",
            "Set up monitoring"
        ]
    }
    
    return creation_result

@app.tool()
async def validate_component_creation(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-validate component creation parameters"""
    pass

@app.tool()
async def get_component_status(component_id: str) -> Dict[str, Any]:
    """Check status of created component"""
    pass
```

### 8. `mcp_servers/registry.py` - MCP Server Registry
**Components:** Server discovery and capability management
```python
"""
MCP Server Registry
Manages discovery of available MCP servers and their capabilities
"""

from fastmcp import FastMCP
from typing import Dict, Any, List

app = FastMCP("MCPRegistry")

@app.tool()
async def discover_mcp_servers() -> Dict[str, Any]:
    """
    Discover available MCP servers and capabilities
    Used in: Orchestrator → MCPRegistry: Discover available servers
    """
    available_servers = {
        "TemplateAgent": {
            "url": "http://localhost:8001/mcp",
            "capabilities": [
                "get_template_params",
                "validate_template_params", 
                "list_available_templates"
            ],
            "supported_templates": ["lambda", "microservice", "database", "frontend"],
            "status": "healthy"
        },
        "CatalogAgent": {
            "url": "http://localhost:8002/mcp",
            "capabilities": [
                "get_developer_context",
                "get_team_info",
                "get_organization_standards"
            ],
            "data_sources": ["backstage-catalog", "ldap", "github"],
            "status": "healthy"
        },
        "ActionAgent": {
            "url": "http://localhost:8003/mcp",
            "capabilities": [
                "create_backstage_component",
                "validate_component_creation",
                "get_component_status"
            ],
            "supported_actions": ["create", "update", "delete", "validate"],
            "status": "healthy"
        }
    }
    
    return available_servers

@app.tool()
async def register_mcp_server(server_info: Dict[str, Any]) -> Dict[str, Any]:
    """Register a new MCP server"""
    pass

@app.tool()
async def health_check_servers() -> Dict[str, List[str]]:
    """Health check all registered servers"""
    pass
```

### 9. `models/workflow_state.py` - State Management
**Components:** Conversation state and data models
```python
"""
Workflow state management and data models
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

class ConversationState(Enum):
    INITIAL = "initial"
    GATHERING_CONTEXT = "gathering_context" 
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    PROCESSING = "processing"
    COMPLETED = "completed"

@dataclass
class WorkflowState:
    """Maintains state throughout the multi-turn conversation"""
    current_state: ConversationState = ConversationState.INITIAL
    user_request: str = ""
    template_type: str = ""
    required_fields: Dict[str, Any] = field(default_factory=dict)
    gathered_context: Dict[str, Any] = field(default_factory=dict)
    user_confirmations: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    session_id: str = "default"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

@dataclass
class TemplateParameter:
    """Template parameter definition"""
    name: str
    type: str
    required: bool = True
    default_value: Any = None
    validation_pattern: Optional[str] = None
    description: str = ""
```

### 10. `models/agent_cards.py` - Agent Card Definitions
**Components:** Agent metadata and capabilities
```python
"""
Agent Card definitions for all agents in the system
"""

from fastmcp.types import AgentCard
from typing import List, Dict, Any

class OrchestrationAgentCard(AgentCard):
    """Agent Card for the Orchestration Host Agent"""
    
    def __init__(self):
        super().__init__(
            name="BackstageTemplateOrchestrator",
            description="Intelligent agent orchestrating Backstage template creation through conversational interaction",
            version="1.0.0",
            author="Agentic Orchestration System",
            capabilities=[
                "Natural language processing of template requests",
                "Multi-turn conversation management", 
                "Intelligent context gathering from MCP servers",
                "Template parameter mapping and validation",
                "Backstage component creation orchestration"
            ],
            supported_protocols=["MCP", "OpenAI"],
            tags=["backstage", "templates", "orchestration", "conversation"],
            metadata={
                "supported_templates": ["lambda", "microservice", "database", "frontend"],
                "mcp_servers": ["TemplateAgent", "CatalogAgent", "ActionAgent"],
                "conversation_states": [state.value for state in ConversationState],
                "llm_integration": "OpenAI ChatGPT",
                "multi_turn_capable": True
            }
        )

# Additional agent cards for other components...
```

### 11. `utils/llm_client.py` - LLM Integration
**Components:** LangChain ChatOpenAI wrapper and utilities
```python
"""
LLM client utilities and integration
"""

import json
import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

class LLMClient:
    """Wrapper for LLM interactions with standardized prompts"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=0.1
        )
    
    async def analyze_user_request(self, user_input: str) -> Dict[str, Any]:
        """Analyze user request to determine template type and requirements"""
        # Implementation of LLM reasoning step from sequence diagram
        pass
    
    async def map_context_to_fields(self, template_info: Dict, catalog_info: Dict, user_request: str) -> Dict[str, Any]:
        """Map gathered context to template fields"""
        # Step 7: Map fields, detect missing/ambiguous
        pass
    
    async def parse_user_corrections(self, user_input: str, current_fields: Dict) -> Dict[str, Any]:
        """Parse user corrections and updates to fields"""
        # Step 9: Parse user confirmations/corrections
        pass
```

### 12. `utils/config.py` - Configuration Management
```python
"""
Configuration management and environment variable handling
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class Config:
    """Application configuration"""
    openai_api_key: str
    openai_model: str = "gpt-4"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    
    # MCP Server endpoints
    template_agent_url: str = "http://localhost:8001"
    catalog_agent_url: str = "http://localhost:8002"  
    action_agent_url: str = "http://localhost:8003"
    registry_url: str = "http://localhost:8004"

def load_config() -> Config:
    """Load configuration from environment variables"""
    load_dotenv()
    
    return Config(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        template_agent_url=os.getenv("TEMPLATE_AGENT_URL", "http://localhost:8001"),
        catalog_agent_url=os.getenv("CATALOG_AGENT_URL", "http://localhost:8002"),
        action_agent_url=os.getenv("ACTION_AGENT_URL", "http://localhost:8003"),
        registry_url=os.getenv("REGISTRY_URL", "http://localhost:8004")
    )
```

## 🚀 Running the System

### Start Individual MCP Servers
```bash
# Terminal 1 - Template Agent  
cd mcp_servers && python -m uvicorn template_agent:app --port 8001

# Terminal 2 - Catalog Agent
cd mcp_servers && python -m uvicorn catalog_agent:app --port 8002

# Terminal 3 - Action Agent  
cd mcp_servers && python -m uvicorn action_agent:app --port 8003

# Terminal 4 - Registry
cd mcp_servers && python -m uvicorn registry:app --port 8004
```

### Start Main Orchestrator
```bash
# Terminal 5 - Main orchestrator service
python main.py

# Or CLI mode for testing
python cli.py
```

## 📋 Requirements

### `requirements.txt`
```txt
fastmcp>=1.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
langchain>=0.1.0
langchain-openai>=0.1.0
python-dotenv>=1.0.0
pydantic>=2.0.0
asyncio-mqtt>=0.13.0
```

### `.env.example`
```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4

# Service Configuration  
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# MCP Server Endpoints
TEMPLATE_AGENT_URL=http://localhost:8001
CATALOG_AGENT_URL=http://localhost:8002
ACTION_AGENT_URL=http://localhost:8003
REGISTRY_URL=http://localhost:8004

# User Context
USER_ID=current_user
```

This modular structure exactly follows your sequence diagram, with each component properly separated and clearly mapped to the flow steps. Each file has a specific responsibility matching the diagram entities.