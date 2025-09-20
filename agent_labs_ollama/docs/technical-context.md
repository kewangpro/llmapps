# Technical Context: Ollama Chat App System Architecture

## Executive Summary

This document provides comprehensive technical specifications for an Ollama-based chat application with integrated agent tool capabilities. The architecture prioritizes local LLM deployment, enterprise-grade security, real-time communication, and scalable tool execution while maintaining privacy-first principles and supporting both individual developers and enterprise environments.

## 1. System Architecture Overview

### 1.1 High-Level Architecture Pattern

**Recommended Pattern**: Decoupled Microservices Architecture

The system employs a four-tier microservices architecture optimized for real-time chat with secure tool integration:

```
┌─────────────────┐    HTTPS/WSS     ┌────────────────────────────┐    HTTP      ┌─────────────────┐
│                 │◄────────────────►│                            │◄────────────►│                 │
│ Frontend Client │                  │ Backend Gateway & Chat     │              │   LLM Service   │
│ (React/Next.js) │                  │ Service (FastAPI/Python)   │              │ (Ollama Proxy)  │
│                 │                  │                            │              │                 │
└─────────────────┘                  └──────────┬─────────────────┘              └─────────────────┘
                                                │ Redis/AMQP
                                                │
                                                ▼
                                    ┌─────────────────────────────┐
                                    │                             │
                                    │ Secure Tool Execution       │
                                    │ Service (Sandboxed Workers) │
                                    │                             │
                                    └─────────────────────────────┘
```

### 1.2 Core Components

#### Frontend Client (SPA)
- **Technology**: React with Next.js and TypeScript
- **Responsibilities**:
  - Real-time chat interface with streaming message display
  - WebSocket connection management for low-latency communication
  - Tool execution status visualization and progress tracking
  - Conversation history management and search capabilities

#### Backend Gateway & Chat Service
- **Technology**: Python with FastAPI (ASGI for async support)
- **Responsibilities**:
  - WebSocket connection termination and session management
  - Request orchestration between LLM and tool execution services
  - Real-time state synchronization and event broadcasting
  - Authentication, authorization, and user session handling

#### LLM Service (Ollama Integration)
- **Technology**: Python or Go with direct Ollama API integration
- **Responsibilities**:
  - Ollama API abstraction and request/response handling
  - Model selection and switching logic
  - Context window management and conversation threading
  - Tool invocation decision making and parameter extraction

#### Secure Tool Execution Service
- **Technology**: Containerized worker pool with message queue
- **Responsibilities**:
  - Sandboxed tool execution in isolated environments
  - Resource monitoring and limit enforcement
  - Security policy validation and audit logging
  - Result processing and error handling

## 2. Ollama API Integration Specifications

### 2.1 Core API Endpoints

#### Chat Completion Endpoint
```http
POST /api/chat
Content-Type: application/json

{
  "model": "llama3.1:8b",
  "messages": [
    {"role": "system", "content": "You are an AI assistant with tool capabilities..."},
    {"role": "user", "content": "Find all Python files in the src directory"}
  ],
  "tools": [...],
  "stream": true
}
```

#### Model Management
```http
# List available models
GET /api/tags

# Pull new model
POST /api/pull
{"name": "llama3.1:8b"}

# Show model information
POST /api/show
{"name": "llama3.1:8b"}

# Create custom model from Modelfile
POST /api/create
{"name": "custom-agent", "modelfile": "FROM llama3.1:8b\nSYSTEM..."}
```

### 2.2 Communication Protocols

#### Streaming Response Handling
```python
import httpx
import json

async def stream_ollama_response(prompt: str, model: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            'POST',
            'http://localhost:11434/api/chat',
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    chunk = json.loads(line)
                    if not chunk.get('done', False):
                        yield chunk['message']['content']
```

#### Error Handling and Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def robust_ollama_call(payload: dict):
    try:
        response = await httpx.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code >= 500:
            raise  # Retry on server errors
        else:
            # Don't retry on client errors
            raise Exception(f"Client error: {e.response.status_code}")
```

### 2.3 Performance Optimization

#### Connection Pooling and Keep-Alive
```python
# Maintain persistent connection pool
OLLAMA_CLIENT = httpx.AsyncClient(
    base_url="http://localhost:11434",
    timeout=httpx.Timeout(30.0),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=5)
)
```

#### Model Preloading and Warming
```python
async def warm_model(model_name: str):
    """Preload model into memory for faster first response"""
    await OLLAMA_CLIENT.post(
        "/api/generate",
        json={
            "model": model_name,
            "prompt": "",
            "keep_alive": "5m"  # Keep model loaded for 5 minutes
        }
    )
```

## 3. Agent Framework Architecture

### 3.1 Tool Registry System

#### Tool Schema Definition
```python
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class ToolSchema(BaseModel):
    name: str
    description: str  # Critical for LLM tool selection
    input_schema: Dict[str, Any]  # JSON Schema
    output_schema: Dict[str, Any]
    security_requirements: List[str]  # e.g., ["filesystem_read", "network_access"]
    resource_limits: Dict[str, Any]  # CPU, memory, timeout limits
    version: str
    author: str

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolSchema] = {}
        self.vector_store = ChromaDB()  # For semantic tool search

    def register_tool(self, tool: ToolSchema):
        self.tools[tool.name] = tool
        # Index tool description for semantic search
        self.vector_store.add_document(
            doc_id=tool.name,
            text=tool.description,
            metadata={"name": tool.name, "capabilities": tool.security_requirements}
        )

    async def find_tools(self, query: str, limit: int = 5) -> List[ToolSchema]:
        """Find tools using semantic search on descriptions"""
        results = await self.vector_store.similarity_search(query, limit=limit)
        return [self.tools[result.doc_id] for result in results]
```

### 3.2 Execution Engine

#### Sandboxed Execution Framework
```python
import docker
import asyncio
from typing import Dict, Any

class SandboxedExecutor:
    def __init__(self):
        self.docker_client = docker.from_env()

    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        security_context: SecurityContext
    ) -> ExecutionResult:

        # Validate security permissions
        await self.security_manager.validate_execution(tool_name, args, security_context)

        # Create isolated container
        container = self.docker_client.containers.run(
            image=f"tools/{tool_name}:latest",
            command=self._build_command(tool_name, args),
            detach=True,
            remove=True,
            network_mode="none",  # No network access by default
            mem_limit="256m",     # Memory limit
            cpu_quota=50000,      # CPU limit (50% of one core)
            read_only=True,       # Read-only filesystem
            user="nobody",        # Non-root user
            security_opt=["no-new-privileges:true"],
            cap_drop=["ALL"]      # Drop all capabilities
        )

        try:
            # Wait for completion with timeout
            result = await asyncio.wait_for(
                self._wait_for_container(container),
                timeout=30.0
            )
            return result
        except asyncio.TimeoutError:
            container.kill()
            raise ToolExecutionTimeout(f"Tool {tool_name} exceeded timeout")
```

### 3.3 Multi-Agent Orchestration

#### Coordinator Pattern Implementation
```python
class AgentCoordinator:
    def __init__(self):
        self.message_bus = RedisMessageBus()
        self.agents: Dict[str, Agent] = {}

    async def execute_workflow(self, workflow: WorkflowDefinition):
        """Execute a multi-step workflow across multiple agents"""
        workflow_state = WorkflowState(workflow.id)

        for step in workflow.steps:
            if step.type == "parallel":
                # Execute multiple agents concurrently
                tasks = [
                    self._execute_agent_task(agent_id, step.task, workflow_state)
                    for agent_id in step.agent_ids
                ]
                results = await asyncio.gather(*tasks)
                workflow_state.add_parallel_results(step.id, results)

            elif step.type == "sequential":
                # Execute agents in sequence
                for agent_id in step.agent_ids:
                    result = await self._execute_agent_task(
                        agent_id, step.task, workflow_state
                    )
                    workflow_state.add_result(step.id, agent_id, result)

        return workflow_state.get_final_result()
```

## 4. Security and Sandboxing Implementation

### 4.1 Defense-in-Depth Security Model

#### Container Security Configuration
```dockerfile
# Minimal security-hardened base image
FROM gcr.io/distroless/python3-debian12

# Create non-root user
USER 65534:65534

# Set security labels
LABEL security.policy="restricted"
LABEL security.capabilities="none"

# Copy only necessary files
COPY --chmod=555 tool_executable /usr/local/bin/

# Set immutable filesystem
VOLUME ["/tmp"]
WORKDIR /tmp

# Default security settings
ENTRYPOINT ["/usr/local/bin/tool_executable"]
```

#### Resource Limits and Monitoring
```python
class ResourceManager:
    def __init__(self):
        self.active_executions: Dict[str, ExecutionContext] = {}

    async def monitor_execution(self, execution_id: str):
        """Monitor resource usage and enforce limits"""
        context = self.active_executions[execution_id]

        while context.is_running:
            stats = await self._get_container_stats(context.container_id)

            if stats.cpu_percent > 80:
                await self._throttle_cpu(context.container_id)

            if stats.memory_usage > context.memory_limit:
                await self._terminate_execution(execution_id, "Memory limit exceeded")
                break

            if context.elapsed_time > context.timeout:
                await self._terminate_execution(execution_id, "Timeout exceeded")
                break

            await asyncio.sleep(1)
```

### 4.2 Policy-Based Access Control

#### Security Policy Engine
```python
class SecurityPolicyEngine:
    def __init__(self):
        self.policies = self._load_policies()

    async def validate_execution(
        self,
        tool_name: str,
        args: Dict[str, Any],
        user_context: UserContext
    ) -> ValidationResult:
        """Validate tool execution against security policies"""

        # Role-based access control
        if not self._check_role_permissions(user_context.role, tool_name):
            return ValidationResult(
                allowed=False,
                reason=f"Role {user_context.role} not permitted to use {tool_name}"
            )

        # Attribute-based access control
        for policy in self.policies:
            if policy.applies_to(tool_name, args):
                if not policy.evaluate(user_context, args):
                    return ValidationResult(
                        allowed=False,
                        reason=policy.denial_reason
                    )

        return ValidationResult(allowed=True)

# Example policy configuration
SECURITY_POLICIES = {
    "file_operations": {
        "allowed_paths": ["/workspace", "/tmp"],
        "denied_patterns": ["*.env", "*.key", "*/secrets/*"],
        "max_file_size": "10MB"
    },
    "network_access": {
        "allowed_domains": ["api.example.com", "*.github.com"],
        "blocked_ports": [22, 3389, 5432],
        "require_approval": True
    }
}
```

### 4.3 Audit Logging and Compliance

#### Comprehensive Audit Trail
```python
class AuditLogger:
    def __init__(self, elasticsearch_client):
        self.es_client = elasticsearch_client

    async def log_execution(
        self,
        execution_id: str,
        user_id: str,
        tool_name: str,
        args: Dict[str, Any],
        result: ExecutionResult,
        security_context: SecurityContext
    ):
        """Log tool execution for audit and compliance"""
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "execution_id": execution_id,
            "user_id": user_id,
            "session_id": security_context.session_id,
            "tool_name": tool_name,
            "tool_args": self._sanitize_args(args),
            "execution_duration_ms": result.duration_ms,
            "exit_code": result.exit_code,
            "resource_usage": {
                "cpu_time_ms": result.cpu_time_ms,
                "memory_peak_mb": result.memory_peak_mb,
                "network_bytes": result.network_bytes
            },
            "security_validations": result.security_checks,
            "compliance_flags": self._check_compliance_requirements(tool_name, args)
        }

        await self.es_client.index(
            index="tool-execution-audit",
            body=audit_record
        )
```

## 5. Technology Stack Recommendations

### 5.1 Core Technology Selection

| Component | Primary Choice | Rationale | Enterprise Alternative |
|-----------|---------------|-----------|----------------------|
| **Backend Language** | Python 3.11+ | Dominant AI/ML ecosystem, extensive LLM libraries | Java/Spring Boot for enterprise standardization |
| **Backend Framework** | FastAPI | High-performance async, native WebSocket support | Django for full-featured enterprise apps |
| **Frontend** | React + Next.js + TypeScript | Industry standard, excellent real-time capabilities | Vue.js + Nuxt.js for simpler learning curve |
| **Database (Primary)** | PostgreSQL 15+ | Reliable, JSONB support for chat history | MongoDB for document-heavy workloads |
| **Vector Database** | ChromaDB / Weaviate | Semantic tool search, RAG capabilities | pgvector for simplified architecture |
| **Message Queue** | Redis + Celery | Proven async task processing | RabbitMQ for complex routing scenarios |
| **Containerization** | Docker + Kubernetes | Enterprise standard, auto-scaling | Serverless (Lambda) for variable workloads |

### 5.2 Development and Deployment Stack

#### Development Environment
```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    volumes:
      - ./backend:/app
      - /var/run/docker.sock:/var/run/docker.sock  # For tool execution
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    ports:
      - "8000:8000"

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    volumes:
      - ./frontend:/app
    ports:
      - "3000:3000"

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: chatapp
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: devpass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
```

#### Production Kubernetes Configuration
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chat-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chat-backend
  template:
    metadata:
      labels:
        app: chat-backend
    spec:
      containers:
      - name: backend
        image: chatapp/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: OLLAMA_BASE_URL
          value: "http://ollama-service:11434"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### 5.3 CI/CD Pipeline

#### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Build and Deploy
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: |
        pytest tests/ --cov=app --cov-report=xml

    - name: Security scan
      run: |
        bandit -r app/
        safety check

    - name: Code quality
      run: |
        black --check app/
        ruff check app/
        mypy app/

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          ghcr.io/${{ github.repository }}/backend:latest
          ghcr.io/${{ github.repository }}/backend:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/chat-backend \
          backend=ghcr.io/${{ github.repository }}/backend:${{ github.sha }}
        kubectl rollout status deployment/chat-backend
```

## 6. Implementation Roadmap

### 6.1 Phase 1: Core Foundation (4-6 weeks)
1. **Basic Chat Interface**: React frontend with WebSocket connection
2. **Ollama Integration**: FastAPI backend with streaming responses
3. **Simple Tool Registry**: In-memory tool storage and basic execution
4. **Container Sandboxing**: Docker-based tool isolation

### 6.2 Phase 2: Security and Scalability (6-8 weeks)
1. **Security Framework**: Policy engine and resource limits
2. **Async Task Processing**: Redis + Celery integration
3. **Vector Database**: ChromaDB for semantic tool search
4. **Audit Logging**: Comprehensive execution tracking

### 6.3 Phase 3: Advanced Features (8-10 weeks)
1. **Multi-Agent Workflows**: Coordinator pattern implementation
2. **Enterprise Features**: SSO, RBAC, compliance reporting
3. **Performance Optimization**: Caching, connection pooling
4. **Monitoring and Observability**: Metrics, tracing, alerting

### 6.4 Phase 4: Enterprise Deployment (4-6 weeks)
1. **Kubernetes Deployment**: Production-ready orchestration
2. **High Availability**: Load balancing, auto-scaling
3. **Backup and Recovery**: Data persistence strategies
4. **Documentation and Training**: User guides, API documentation

## 7. Performance and Scalability Considerations

### 7.1 Real-Time Communication Scaling

#### WebSocket Connection Management
```python
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.redis_client = redis.Redis()

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        # Subscribe to user-specific Redis channel
        await self.redis_client.subscribe(f"user:{user_id}")

    async def broadcast_to_user(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            # Direct connection - send immediately
            await self.active_connections[user_id].send_json(message)
        else:
            # User connected to different instance - use Redis
            await self.redis_client.publish(f"user:{user_id}", json.dumps(message))
```

### 7.2 Horizontal Scaling Strategy

#### Load Balancing Configuration
```nginx
# nginx.conf for WebSocket load balancing
upstream chat_backend {
    ip_hash;  # Sticky sessions for WebSocket connections
    server backend-1:8000;
    server backend-2:8000;
    server backend-3:8000;
}

server {
    listen 80;

    location /ws {
        proxy_pass http://chat_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

### 7.3 Performance Monitoring

#### Application Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
TOOL_EXECUTIONS = Counter('tool_executions_total', 'Total tool executions', ['tool_name', 'status'])
EXECUTION_DURATION = Histogram('tool_execution_duration_seconds', 'Tool execution time')
ACTIVE_CONNECTIONS = Gauge('websocket_connections_active', 'Active WebSocket connections')
LLM_RESPONSE_TIME = Histogram('llm_response_time_seconds', 'LLM response time')

# Usage in application
@EXECUTION_DURATION.time()
async def execute_tool(tool_name: str, args: dict):
    try:
        result = await tool_executor.run(tool_name, args)
        TOOL_EXECUTIONS.labels(tool_name=tool_name, status='success').inc()
        return result
    except Exception as e:
        TOOL_EXECUTIONS.labels(tool_name=tool_name, status='error').inc()
        raise
```

## Conclusion

This technical architecture provides a robust foundation for an enterprise-ready Ollama chat application with integrated tool capabilities. The design prioritizes security, scalability, and maintainability while leveraging the unique advantages of local LLM deployment. The recommended technology stack balances cutting-edge AI capabilities with proven enterprise technologies, ensuring both innovation and reliability.

The implementation approach follows a phased roadmap that allows for iterative development and validation, enabling rapid prototyping while building toward a production-ready system that can scale from individual developer use cases to enterprise-wide deployments.
