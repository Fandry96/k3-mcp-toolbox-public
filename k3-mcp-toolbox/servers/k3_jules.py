#!/usr/bin/env python3
"""
k3-jules — Google Jules Autonomous Engineering Runtime MCP Server
Provides headless orchestration of Google Jules cloud VM sandboxes via
the v1alpha REST API (https://jules.googleapis.com/v1alpha).
Supports repoless sandboxes (Node.js, Python, Rust, Bun), Git-bound PR generation,
structured ChangeSet patch downloads, and RFC 3339 cursor-based activity polling.
"""

import os
import json
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional, Dict, Any, List

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server with Pydantic 2.x safety guard
mcp = FastMCP("k3-jules")
mcp._mcp_server.version = "1.0.0"

API_BASE = "https://jules.googleapis.com/v1alpha"


def _get_api_key() -> str:
    """Retrieves Jules API key from environment."""
    key = os.environ.get("JULES_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        # Check .env.local in K3_Firehose root as fallback
        env_path = r"c:\K3_Firehose\.env.local"
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("JULES_API_KEY="):
                        return line.split("=", 1)[1].strip("'\"")
                    if not key and line.startswith("GEMINI_API_KEY="):
                        key = line.split("=", 1)[1].strip("'\"")
    return key or ""


def _http_request(
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Makes an authenticated HTTP request to the Jules REST API."""
    api_key = _get_api_key()
    if not api_key:
        return {
            "error": "Missing JULES_API_KEY or GEMINI_API_KEY in environment or .env.local"
        }

    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    if params:
        query_string = urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None}
        )
        if query_string:
            url += f"?{query_string}"

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
        "User-Agent": "K3-Antigravity-MCP/1.0",
    }

    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content) if content else {"status": "ok"}
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            return {"error": f"HTTP {e.code}: {e.reason}", "details": json.loads(error_body)}
        except Exception:
            return {"error": f"HTTP {e.code}: {e.reason}", "raw_body": error_body}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def jules_list_sources(page_size: int = 20, page_token: Optional[str] = None) -> str:
    """
    Lists all authorized GitHub repositories connected to the Google Jules profile.
    Returns repository owner, name, default branch, and visibility.
    """
    params = {"pageSize": page_size}
    if page_token:
        params["pageToken"] = page_token

    res = _http_request("sources", method="GET", params=params)
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_get_source(source_id: str) -> str:
    """
    Retrieves structural metadata and branch trees for an authorized repository.
    Args:
        source_id: Resource path or ID (e.g. 'sources/github/Fandry96/realtor-web' or 'github/Fandry96/realtor-web')
    """
    clean_id = source_id if source_id.startswith("sources/") else f"sources/{source_id}"
    res = _http_request(clean_id, method="GET")
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_create_session(
    prompt: str,
    source_name: Optional[str] = None,
    branch: Optional[str] = None,
    title: Optional[str] = None,
    auto_create_pr: bool = True,
    require_plan_approval: bool = False,
) -> str:
    """
    Provisions an isolated cloud sandbox and initiates an asynchronous Jules coding session.
    If source_name is omitted, Jules provisions an ephemeral REPOLESS sandbox (Node.js, Python, Rust, Bun).
    If source_name is provided, Jules clones the repo, plans changes, executes tests, and delivers PR/patches.

    Args:
        prompt: Task instructions or code prompt for the agent.
        source_name: Optional repository resource name (e.g. 'sources/github/Fandry96/realtor-web'). Omit for repoless.
        branch: Target Git branch (default: repo default branch).
        title: Descriptive title for tracking in Jules dashboard/CLI.
        auto_create_pr: If True, automatically creates a GitHub Pull Request upon completion.
        require_plan_approval: If True, pauses in AWAITING_PLAN_APPROVAL until jules_approve_plan is called.
    """
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "requirePlanApproval": require_plan_approval,
        "automationMode": "AUTO_CREATE_PR" if auto_create_pr else "AUTOMATION_MODE_UNSPECIFIED",
    }
    if title:
        payload["title"] = title
    if source_name:
        clean_source = source_name if source_name.startswith("sources/") else f"sources/{source_name}"
        context: Dict[str, Any] = {"source": clean_source}
        if branch:
            context["branch"] = branch
        payload["sourceContext"] = context

    res = _http_request("sessions", method="POST", body=payload)
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_list_sessions(page_size: int = 20, page_token: Optional[str] = None) -> str:
    """
    Returns a paginated index of active and historical Jules compute sessions.
    Shows session state (QUEUED, PLANNING, IN_PROGRESS, COMPLETED, FAILED), URLs, and PR links.
    """
    params = {"pageSize": page_size}
    if page_token:
        params["pageToken"] = page_token

    res = _http_request("sessions", method="GET", params=params)
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_get_session(session_id: str) -> str:
    """
    Polls the runtime state, lifecycle phase, and PR output links for a specific Jules session.
    Args:
        session_id: Session identifier (e.g. 'sessions/12345678' or '12345678')
    """
    clean_id = session_id if session_id.startswith("sessions/") else f"sessions/{session_id}"
    res = _http_request(clean_id, method="GET")
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_approve_plan(session_id: str) -> str:
    """
    Approves an implementation plan when requirePlanApproval was enabled,
    transitioning the session state from AWAITING_PLAN_APPROVAL to IN_PROGRESS.
    """
    clean_id = session_id if session_id.startswith("sessions/") else f"sessions/{session_id}"
    endpoint = f"{clean_id}:approvePlan"
    res = _http_request(endpoint, method="POST", body={})
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_send_message(session_id: str, prompt: str) -> str:
    """
    Injects steering guidance, corrections, or prompt adjustments into an active Jules session.
    Args:
        session_id: Session identifier.
        prompt: Message or steering instructions to send to the running agent.
    """
    clean_id = session_id if session_id.startswith("sessions/") else f"sessions/{session_id}"
    endpoint = f"{clean_id}:sendMessage"
    res = _http_request(endpoint, method="POST", body={"prompt": prompt})
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_list_activities(
    session_id: str,
    create_time_cursor: Optional[str] = None,
    page_size: int = 50,
    page_token: Optional[str] = None,
) -> str:
    """
    Queries the event-sourced activity log and artifact payloads of a Jules session.
    Use create_time_cursor (RFC 3339 timestamp) for delta stream polling to retrieve
    only events emitted after the specified timestamp.
    """
    clean_id = session_id if session_id.startswith("sessions/") else f"sessions/{session_id}"
    endpoint = f"{clean_id}/activities"
    params: Dict[str, Any] = {"pageSize": page_size}
    if create_time_cursor:
        params["createTime"] = create_time_cursor
    if page_token:
        params["pageToken"] = page_token

    res = _http_request(endpoint, method="GET", params=params)
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_get_patch(session_id: str) -> str:
    """
    Extracts the structured ChangeSet patch artifact from a completed Jules session.
    Returns the unified diff (unidiffPatch), baseCommitId, and suggestedCommitMessage
    allowing programmatic local patch application via `git apply` without creating Git branches.
    """
    clean_id = session_id if session_id.startswith("sessions/") else f"sessions/{session_id}"
    endpoint = f"{clean_id}/activities"
    res = _http_request(endpoint, method="GET", params={"pageSize": 100})

    if "error" in res:
        return json.dumps(res, indent=2)

    activities = res.get("activities", [])
    patches = []

    for act in activities:
        artifacts = act.get("artifacts", [])
        for art in artifacts:
            cs = art.get("changeSet")
            if cs:
                patches.append(
                    {
                        "activityId": act.get("id"),
                        "originator": act.get("originator"),
                        "baseCommitId": cs.get("baseCommitId"),
                        "suggestedCommitMessage": cs.get("suggestedCommitMessage"),
                        "unidiffPatch": cs.get("unidiffPatch"),
                    }
                )

    if not patches:
        return json.dumps(
            {
                "status": "no_patch_found",
                "message": "No structured ChangeSet artifact has been emitted yet for this session.",
                "total_activities_scanned": len(activities),
            },
            indent=2,
        )

    return json.dumps({"status": "ok", "patches": patches}, indent=2)


if __name__ == "__main__":
    mcp.run()
