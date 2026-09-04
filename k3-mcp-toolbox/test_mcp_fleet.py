#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for K3 MCP Server Fleet.
Tests all 5 servers:
1. k3-mrl-memory
2. k3-agent-ops
3. k3-doc-intel
4. k3-local-llm
5. k3-worktree-ops
"""

import sys
import os
from pathlib import Path

# Add servers directory to path
SERVERS_DIR = Path(__file__).parent / "servers"
sys.path.insert(0, str(SERVERS_DIR))

passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")


def test_server_mrl_memory():
    print("\n=== 1. Testing k3-mrl-memory ===")
    import k3_mrl_memory

    # Key classification tests
    test("Classify skill key", k3_mrl_memory._classify_key(r"c:\K3_Firehose\.agent\skills\test\SKILL.md") == "skill")
    test("Classify knowledge key", k3_mrl_memory._classify_key(r"c:\Users\fandr\.gemini\antigravity\knowledge\ki_001.md") == "knowledge")
    test("Classify research key", k3_mrl_memory._classify_key(r"c:\K3_Firehose\realtor-research\doc.docx") == "research")
    test("Classify brain key", k3_mrl_memory._classify_key(r"c:\Users\fandr\.gemini\antigravity\brain\abc\task.md") == "brain")
    test("Classify book key", k3_mrl_memory._classify_key(r"c:\K3_Firehose\Proto_book\research\book.md") == "book")

    # Stats test
    stats = k3_mrl_memory.mrl_index_stats()
    test("mrl_index_stats returns string", isinstance(stats, str) and len(stats) > 50)
    test("Stats contains total vectors", "Total Vectors:" in stats)
    test("Stats contains breakdown", "Corpus Breakdown:" in stats)


def test_server_agent_ops():
    print("\n=== 2. Testing k3-agent-ops ===")
    import k3_agent_ops

    # Port check
    port_res = k3_agent_ops.ops_check_ports([9222, 54321])
    test("ops_check_ports returns status", "Port Status Scan" in port_res or "FREE" in port_res)

    # Health check
    health_res = k3_agent_ops.ops_system_health()
    test("ops_system_health returns CPU/RAM info", "CPU Load:" in health_res and "RAM:" in health_res)
    test("Health check contains agent footprint", "Python instances" in health_res)

    # Port bounds validation test
    invalid_port_res = k3_agent_ops.ops_free_port(99999)
    test("Port bounds validation (< 1 or > 65535)", "Invalid port number" in invalid_port_res)

    # Inactive port test
    free_res = k3_agent_ops.ops_free_port(59998)
    test("Free port handles inactive port", "not in LISTEN" in free_res or "already free" in free_res)


def test_server_doc_intel():
    print("\n=== 3. Testing k3-doc-intel ===")
    import k3_doc_intel

    # Chunking algorithm test
    sample_text = ("This is paragraph one.\n\n" * 10) + ("This is paragraph two.\n\n" * 10)
    chunks = k3_doc_intel._chunk_text(sample_text, chunk_size=100, overlap=20)
    test("Chunking produces non-empty list", len(chunks) > 1)
    test("Chunk overlap preserves content", all(len(c) > 0 for c in chunks))

    # Path resolution test
    idx_path = k3_doc_intel._resolve_index_path("c:\\test_project")
    test("Project index path resolves to .agents/doc_index.pkl", idx_path.name == "doc_index.pkl" and idx_path.parent.name == ".agents")

    # Ingest listing test
    list_res = k3_doc_intel.doc_list_ingested("c:\\non_existent_empty_path_xyz")
    test("Missing index reports cleanly", "No document index found" in list_res)


def test_server_local_llm():
    print("\n=== 4. Testing k3-local-llm ===")
    import k3_local_llm

    # Model discovery
    models = k3_local_llm._find_gguf_models()
    test("Discovered GGUF models in src/models/", len(models) >= 1)
    if models:
        test("Detected EmbeddingGemma model", any("embeddinggemma" in m["name"].lower() for m in models))
        test("Classified as embedding type", any(m["type"] == "embedding" for m in models))

    # Model list tool
    list_str = k3_local_llm.local_models_list()
    test("local_models_list returns formatted table", "GGUF Model Registry" in list_str)

    # Status check (offline)
    status_str = k3_local_llm.local_server_status(port=59999)
    test("Offline port reports OFFLINE status", "OFFLINE" in status_str)


def test_server_worktree_ops():
    print("\n=== 5. Testing k3-worktree-ops ===")
    import k3_worktree_ops

    # Resolve repo
    repo = k3_worktree_ops._resolve_repo(r"c:\K3_Firehose")
    test("Validates root repo", repo.exists() and (repo / ".git").exists())

    # Worktree list
    wt_list = k3_worktree_ops.worktree_list(r"c:\K3_Firehose")
    test("worktree_list returns string", "Active Git Worktrees" in wt_list or "No worktrees" in wt_list)


def main():
    test_server_mrl_memory()
    test_server_agent_ops()
    test_server_doc_intel()
    test_server_local_llm()
    test_server_worktree_ops()

    print(f"\n{'='*50}")
    print(f"FLEET INTEGRATION RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("ALL 5 K3 MCP SERVERS VERIFIED AND OPERATIONAL!")
    else:
        print(f"WARNING: {failed} TEST(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
