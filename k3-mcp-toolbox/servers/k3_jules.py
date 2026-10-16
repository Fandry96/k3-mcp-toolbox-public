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
    """Retrieves Jules API key from environment, .env.local, or GCP Secret Manager."""
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
        if key:
            return key
        # GCP Secret Manager fallback
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            name = "projects/gen-lang-client-0778100894/secrets/JULES_API_KEY/versions/latest"
            resp = client.access_secret_version(request={"name": name})
            return resp.payload.data.decode("utf-8").strip()
        except Exception:
            pass
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
    """List authorized GitHub repositories connected to Google Jules with owner, default branch, and visibility. Use INSTEAD OF curl/REST scripts or manual GitHub App checks when discovering valid source_name targets for Jules sessions. Do NOT use for: general GitHub repo search outside Jules (use github-mcp-server).
    Args:
        page_size: Max results (default: 20).
        page_token: Pagination cursor.
    """
    params = {"pageSize": page_size}
    if page_token:
        params["pageToken"] = page_token

    res = _http_request("sources", method="GET", params=params)
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_get_source(source_id: str) -> str:
    """Retrieve branch metadata and repository structure for a single authorized Google Jules GitHub source. Use INSTEAD OF raw curl calls to jules.googleapis.com when verifying branch names before launching a session. Do NOT use for: reading file contents inside a repository (use github-mcp-server or view_file).
    Args:
        source_id: Jules source ID (e.g. 'sources/github/Fandry96/realtor-web').
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
    """Launch a Google Jules cloud coding sandbox (repo PR or repoless VM). Use INSTEAD OF issue bots or curl POSTs to delegate autonomous coding and PRs to Jules. Do NOT use for: local workspace tasks (use invoke_subagent).
    Args:
        prompt: Task instructions.
        source_name: Repo (omit = repoless).
        branch: Target branch.
        title: Session title.
        auto_create_pr: Open PR when done.
        require_plan_approval: Pause for jules_approve_plan.
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
    """List active and historical Google Jules sessions with lifecycle states and PR links. Use INSTEAD OF manual browser dashboard checks or curl scripts when enumerating running Jules jobs. Do NOT use for: polling a single known session ID (use jules_get_session) or local background tasks (use manage_task).
    Args:
        page_size: Max sessions to return (default: 20).
        page_token: Pagination token.
    """
    params = {"pageSize": page_size}
    if page_token:
        params["pageToken"] = page_token

    res = _http_request("sessions", method="GET", params=params)
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_get_session(session_id: str) -> str:
    """Poll the lifecycle state, phase, and generated PR URL for a specific Google Jules session. Use INSTEAD OF raw curl polling loops against jules.googleapis.com when checking whether a Jules job finished or needs approval. Do NOT use for: streaming step-by-step activity logs (use jules_list_activities).
    Args:
        session_id: Jules session ID (e.g. 'sessions/12345678' or '12345678').
    """
    clean_id = session_id if session_id.startswith("sessions/") else f"sessions/{session_id}"
    res = _http_request(clean_id, method="GET")
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_approve_plan(session_id: str) -> str:
    """Approve a pending implementation plan for a Google Jules session in AWAITING_PLAN_APPROVAL state. Use INSTEAD OF manual web UI clicks or raw REST POSTs when advancing a gated Jules session to execution. Do NOT use for: sending text revisions or steering feedback (use jules_send_message).
    Args:
        session_id: Jules session ID awaiting plan approval.
    """
    clean_id = session_id if session_id.startswith("sessions/") else f"sessions/{session_id}"
    endpoint = f"{clean_id}:approvePlan"
    res = _http_request(endpoint, method="POST", body={})
    return json.dumps(res, indent=2)


@mcp.tool()
def jules_send_message(session_id: str, prompt: str) -> str:
    """Send steering instructions or follow-up corrections into an active Google Jules session. Use INSTEAD OF GitHub PR comment triggers or manual web UI chat when redirecting an in-flight Jules sandbox. Do NOT use for: messaging local Antigravity subagents (use send_message).
    Args:
        session_id: Jules session ID.
        prompt: Steering instructions for the running Jules agent.
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
    """Query the event-sourced activity log and artifacts of a Google Jules session, optionally filtered by an RFC 3339 cursor. Use INSTEAD OF raw curl pagination when inspecting sandbox steps or errors. Do NOT use for: ready-to-apply git diffs (use jules_get_patch).
    Args:
        session_id: Jules session ID.
        create_time_cursor: RFC 3339 timestamp filter.
        page_size: Max events (default: 50).
        page_token: Pagination cursor.
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
    """Extract the unified diff (unidiffPatch), baseCommitId, and commit message from a completed Google Jules session. Use INSTEAD OF manually parsing activity JSON blobs when applying repoless or unmerged Jules changes locally via git apply. Do NOT use for: checking high-level session status (use jules_get_session).
    Args:
        session_id: Jules session ID.
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
