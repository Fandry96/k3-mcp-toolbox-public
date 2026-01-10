# Docker MCP Gateway

This directory contains a **Reference Architecture** for running Model Context Protocol (MCP) servers within a Dockerized environment.

## The Concept

The **Docker MCP Gateway** allows you to:

1. Run MCP servers (like `fetch`, `filesystem`, or custom tools) inside isolated containers.
2. Expose them to your primary Agent via a standardized HTTP Gateway.
3. Scale your toolset dynamically without polluting your host OS.

## Prerequisites

* Docker Desktop / Docker Engine
* (Optional) Docker MCP Extension (if using managed plugins)

## Usage

1. **Start the Gateway:**

    ```bash
    docker-compose up -d
    ```

2. **Connect your Agent:**
    Configure your agent (e.g., Claude Desktop, Zed, or custom Python script) to connect to `http://localhost:8000/sse` (or correct endpoint depending on gateway version).

## Configuration

Edit the `command` in `docker-compose.yml` to enable different servers:

```yaml
command: >
  docker mcp gateway run
    --servers=duckduckgo,filesystem,postgres
```

## Security Note

Mounting `/var/run/docker.sock` gives the container full control over your Docker daemon. Only use this in trusted environments or use a secure proxy.
