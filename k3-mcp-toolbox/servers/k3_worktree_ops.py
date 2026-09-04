#!/usr/bin/env python3
"""
k3-worktree-ops — Git Worktree & Branch Orchestrator MCP Server
Automates the Antigravity Worktree Isolation Protocol.
Enables subagents to create, inspect, diff, verify, and merge isolated git worktrees
under ~/.gemini/antigravity/worktrees/ to eliminate Windows file-locking collisions.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("k3-worktree-ops")

DEFAULT_REPO = Path(r"C:\K3_Firehose").resolve()
WORKTREES_ROOT = Path.home() / ".gemini" / "antigravity" / "worktrees"


def _run_git(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    """Executes a git command with UTF-8 encoding."""
    return subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def _resolve_repo(repo_path: Optional[str] = None) -> Path:
    p = Path(repo_path).resolve() if repo_path else DEFAULT_REPO
    if not (p / ".git").exists() and not (p / ".git").is_file():
        raise ValueError(f"Directory {p} is not a valid git repository.")
    return p


@mcp.tool()
def worktree_list(repo_path: Optional[str] = None) -> str:
    """
    Lists all active Git worktrees for the repository, showing their directory path,
    HEAD commit hash, and checked-out branch.
    """
    try:
        repo = _resolve_repo(repo_path)
    except Exception as e:
        return f"Error: {e}"

    res = _run_git(["worktree", "list", "--porcelain"], cwd=repo)
    if res.returncode != 0:
        return f"git worktree list failed: {res.stderr.strip()}"

    raw = res.stdout.strip()
    if not raw:
        return "No worktrees registered."

    # Parse porcelain format
    worktrees = []
    current: Dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            if current:
                worktrees.append(current)
                current = {}
            continue
        parts = line.split(" ", 1)
        key = parts[0]
        val = parts[1] if len(parts) > 1 else ""
        current[key] = val
    if current:
        worktrees.append(current)

    lines = [
        f"=== Active Git Worktrees ===",
        f"Repository: {repo}",
        f"Total Worktrees: {len(worktrees)}\n",
    ]
    for i, wt in enumerate(worktrees, 1):
        path = wt.get("worktree", "unknown")
        head = wt.get("HEAD", "")[:8]
        branch = wt.get("branch", "").replace("refs/heads/", "") or "detached HEAD"
        is_bare = "bare" in wt
        status = "(Bare root)" if is_bare else f"Branch: {branch} [{head}]"
        lines.append(f"{i}. {path}\n   {status}")

    return "\n".join(lines)


@mcp.tool()
def worktree_create(
    branch_name: str,
    base_commit: str = "HEAD",
    repo_path: Optional[str] = None
) -> str:
    """
    Provisions a new isolated Git worktree under ~/.gemini/antigravity/worktrees/{branch_name}.
    Guarantees isolation from the main workspace to prevent Windows file-lock conflicts.
    Args:
        branch_name: New git branch name for the worktree.
        base_commit: Starting branch or commit (default: HEAD).
        repo_path: Root repo directory (default: C:\\K3_Firehose).
    """
    try:
        repo = _resolve_repo(repo_path)
    except Exception as e:
        return f"Error: {e}"

    clean_branch = branch_name.strip().replace(" ", "-")
    target_dir = WORKTREES_ROOT / clean_branch
    WORKTREES_ROOT.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        return f"Error: Worktree path already exists at {target_dir}. Use worktree_cleanup() first."

    # Check if branch exists
    branch_check = _run_git(["rev-parse", "--verify", f"refs/heads/{clean_branch}"], cwd=repo)
    if branch_check.returncode == 0:
        # Branch exists, attach to existing branch
        cmd = ["worktree", "add", str(target_dir), clean_branch]
    else:
        # Create new branch
        cmd = ["worktree", "add", "-b", clean_branch, str(target_dir), base_commit]

    res = _run_git(cmd, cwd=repo)
    if res.returncode != 0:
        return f"Failed to create worktree: {res.stderr.strip()}"

    # Get HEAD commit
    head_res = _run_git(["rev-parse", "--short", "HEAD"], cwd=target_dir)
    head_sha = head_res.stdout.strip() if head_res.returncode == 0 else "unknown"

    return (
        f"=== Isolated Worktree Created ===\n"
        f"Branch Name  : {clean_branch}\n"
        f"Base Commit  : {base_commit} (Current HEAD: {head_sha})\n"
        f"Worktree Dir : {target_dir}\n"
        f"Status       : Ready for parallel subagent execution without lock conflicts.\n\n"
        f"Instructions: Point your subagent workspace to: {target_dir}"
    )


@mcp.tool()
def worktree_diff(
    branch_name: str,
    target_branch: str = "main",
    repo_path: Optional[str] = None
) -> str:
    """
    Computes git diff statistics and file change summaries between worktree branch and target.
    Args:
        branch_name: Feature branch in the worktree.
        target_branch: Comparison branch (default: main).
    """
    try:
        repo = _resolve_repo(repo_path)
    except Exception as e:
        return f"Error: {e}"

    # Check stat
    stat_res = _run_git(["diff", "--stat", f"{target_branch}...{branch_name}"], cwd=repo)
    if stat_res.returncode != 0:
        # Try without 3 dots
        stat_res = _run_git(["diff", "--stat", f"{target_branch}..{branch_name}"], cwd=repo)
        if stat_res.returncode != 0:
            return f"Diff failed: {stat_res.stderr.strip()}"

    # Name-status
    name_res = _run_git(["diff", "--name-status", f"{target_branch}...{branch_name}"], cwd=repo)

    lines = [
        f"=== Git Worktree Diff: {branch_name} vs {target_branch} ===",
        f"Repository: {repo}\n",
        "Summary Statistics:",
        stat_res.stdout.strip() or "No file differences detected.",
        "\nChanged Files:",
        name_res.stdout.strip() or "None",
    ]
    return "\n".join(lines)


@mcp.tool()
def worktree_cleanup(
    branch_name: str,
    force: bool = False,
    delete_branch: bool = True,
    repo_path: Optional[str] = None
) -> str:
    """
    Removes a Git worktree and optionally deletes the associated feature branch.
    Args:
        branch_name: Branch name to tear down.
        force: Force removal even if uncommitted changes exist.
        delete_branch: Also delete the git branch (default: True).
    """
    try:
        repo = _resolve_repo(repo_path)
    except Exception as e:
        return f"Error: {e}"

    clean_branch = branch_name.strip()
    target_dir = WORKTREES_ROOT / clean_branch

    # Run git worktree remove
    cmd = ["worktree", "remove", str(target_dir)]
    if force:
        cmd.append("--force")

    res = _run_git(cmd, cwd=repo)
    msg = []
    if res.returncode == 0:
        msg.append(f"Worktree directory removed at {target_dir}.")
    else:
        # If directory was manually removed or pruned, run worktree prune
        _run_git(["worktree", "prune"], cwd=repo)
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
            msg.append(f"Worktree pruned and purged manually: {target_dir}.")
        else:
            msg.append(f"Worktree pruned from git registry.")

    if delete_branch:
        del_flag = "-D" if force else "-d"
        del_res = _run_git(["branch", del_flag, clean_branch], cwd=repo)
        if del_res.returncode == 0:
            msg.append(f"Branch '{clean_branch}' deleted successfully.")
        else:
            msg.append(f"Branch delete note: {del_res.stderr.strip() or 'branch retained'}")

    return "=== Worktree Cleanup ===\n" + "\n".join(msg)


@mcp.tool()
def worktree_merge_and_cleanup(
    branch_name: str,
    target_branch: str = "main",
    verify_command: Optional[str] = None,
    repo_path: Optional[str] = None
) -> str:
    """
    Verifies changes inside the worktree, merges them into the target branch, and cleans up.
    If verify_command fails, the merge is completely aborted.
    Args:
        branch_name: Feature branch to merge.
        target_branch: Destination branch (default: main).
        verify_command: Optional shell command to run in worktree before merging (e.g. 'python -m pytest').
    """
    try:
        repo = _resolve_repo(repo_path)
    except Exception as e:
        return f"Error: {e}"

    clean_branch = branch_name.strip()
    target_dir = WORKTREES_ROOT / clean_branch

    # 1. Run Pre-Merge Verification
    if verify_command:
        if not target_dir.exists():
            return f"Error: Worktree directory not found at {target_dir}."
        v_res = subprocess.run(
            verify_command,
            shell=True,
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if v_res.returncode != 0:
            return (
                f"ABORTED: Pre-merge verification failed with exit code {v_res.returncode}!\n"
                f"Command: {verify_command}\n\n"
                f"Stdout:\n{v_res.stdout[:500]}\n\n"
                f"Stderr:\n{v_res.stderr[:500]}\n\n"
                f"Fix the issues in {target_dir} before merging."
            )

    # 2. Checkout target branch in root repo
    co_res = _run_git(["checkout", target_branch], cwd=repo)
    if co_res.returncode != 0:
        return f"Failed to checkout {target_branch}: {co_res.stderr.strip()}"

    # 3. Merge feature branch
    merge_res = _run_git(["merge", clean_branch, "--no-ff", "-m", f"Merge worktree branch '{clean_branch}'"], cwd=repo)
    if merge_res.returncode != 0:
        return (
            f"Merge conflict or failure merging '{clean_branch}' into '{target_branch}':\n"
            f"{merge_res.stderr.strip() or merge_res.stdout.strip()}"
        )

    # 4. Cleanup worktree
    cleanup_res = worktree_cleanup(clean_branch, force=False, delete_branch=True, repo_path=str(repo))

    return (
        f"=== Worktree Successfully Merged & Closed ===\n"
        f"Source Branch : {clean_branch}\n"
        f"Target Branch : {target_branch}\n"
        f"Merge Output  : {merge_res.stdout.strip()}\n\n"
        f"{cleanup_res}"
    )


if __name__ == "__main__":
    mcp.run()
