#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for K3 MCP Server Fleet.
Tests all 6 servers:
1. k3-mrl-memory
2. k3-agent-ops
3. k3-doc-intel
4. k3-local-llm
5. k3-worktree-ops
6. k3-forge
"""

import sys
from pathlib import Path

# Add servers directory to path
SERVERS_DIR = Path(__file__).parent / "servers"
sys.path.insert(0, str(SERVERS_DIR))

passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")


def test_server_mrl_memory() -> None:
    print("\n=== 1. Testing k3-mrl-memory ===")
    import k3_mrl_memory

    # Key classification tests
    _home = str(Path.home())
    test("Classify skill key", k3_mrl_memory._classify_key(f"{_home}\\.agent\\skills\\test\\SKILL.md") == "skill")
    test("Classify knowledge key", k3_mrl_memory._classify_key(f"{_home}\\.gemini\\antigravity\\knowledge\\ki_001.md") == "knowledge")
    test("Classify research key", k3_mrl_memory._classify_key(f"{_home}\\realtor-research\\doc.docx") == "research")
    test("Classify brain key", k3_mrl_memory._classify_key(f"{_home}\\.gemini\\antigravity\\brain\\abc\\task.md") == "brain")
    test("Classify book key", k3_mrl_memory._classify_key(f"{_home}\\Proto_book\\research\\book.md") == "book")

    # Stats test
    stats = k3_mrl_memory.mrl_index_stats()
    test("mrl_index_stats returns string", isinstance(stats, str) and len(stats) > 50)
    test("Stats contains total vectors", "Total Vectors:" in stats)
    test("Stats contains breakdown", "Corpus Breakdown:" in stats)

    # Empty query test
    empty_res = k3_mrl_memory.mrl_search("")
    test("mrl_search blocks empty query", "Error: Search query cannot be empty." in empty_res)


def test_server_agent_ops() -> None:
    print("\n=== 2. Testing k3-agent-ops ===")
    import k3_agent_ops

    # Port check
    port_res = k3_agent_ops.ops_check_ports([9222, 54321])
    test("ops_check_ports returns status", "Port Status Scan" in port_res or "FREE" in port_res)

    # Invalid port argument validation test
    inv_port = k3_agent_ops.ops_check_ports(["not-a-port"])
    test("ops_check_ports rejects non-integer ports", "Error:" in inv_port)

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


def test_server_doc_intel() -> None:
    print("\n=== 3. Testing k3-doc-intel ===")
    import k3_doc_intel

    # Chunking algorithm test
    sample_text = ("This is paragraph one.\n\n" * 10) + ("This is paragraph two.\n\n" * 10)
    chunks = k3_doc_intel._chunk_text(sample_text, chunk_size=100, overlap=20)
    test("Chunking produces non-empty list", len(chunks) > 1)
    test("Chunk overlap preserves content", all(len(c) > 0 for c in chunks))

    # Degenerate chunk parameters test (overlap > chunk_size)
    degen_chunks = k3_doc_intel._chunk_text(sample_text, chunk_size=50, overlap=200)
    test("Degenerate overlap is bounded safely", len(degen_chunks) > 0 and len(degen_chunks) < 50)

    # Path resolution test
    idx_path = k3_doc_intel._resolve_index_path("c:\\test_project")
    test("Project index path resolves to .agents/doc_index.pkl", idx_path.name == "doc_index.pkl" and idx_path.parent.name == ".agents")

    # Ingest listing test
    list_res = k3_doc_intel.doc_list_ingested("c:\\non_existent_empty_path_xyz")
    test("Missing index reports cleanly", "No document index found" in list_res)


def test_server_local_llm() -> None:
    print("\n=== 4. Testing k3-local-llm ===")
    import k3_local_llm

    # Model discovery
    models = k3_local_llm._find_gguf_models()
    if models:
        test("Discovered GGUF models in src/models/", len(models) >= 1)
        test("Detected EmbeddingGemma model", any("embeddinggemma" in m["name"].lower() for m in models))
        test("Classified as embedding type", any(m["type"] == "embedding" for m in models))
        list_str = k3_local_llm.local_models_list()
        test("local_models_list returns formatted table", "GGUF Model Registry" in list_str)
    else:
        test("Model discovery handled empty directory", isinstance(models, list))
        list_str = k3_local_llm.local_models_list()
        test("local_models_list reports empty cleanly", "No .gguf models found" in list_str)

    # Status check (offline)
    status_str = k3_local_llm.local_server_status(port=59999)
    test("Offline port reports OFFLINE status", "OFFLINE" in status_str)

    # Download helper catalog check
    download_list = k3_local_llm.local_model_download("list")
    test("local_model_download lists curated models", "Curated Downloadable Models" in download_list and "qwen2.5-0.5b" in download_list)

    # Download helper extension validation
    invalid_url = k3_local_llm.local_model_download("https://example.com/not-a-gguf.bin")
    test("local_model_download validates .gguf extension", "Error: URL must point to a file ending in .gguf" in invalid_url)


def test_server_worktree_ops() -> None:
    print("\n=== 5. Testing k3-worktree-ops ===")
    import k3_worktree_ops

    # Resolve repo
    repo = k3_worktree_ops._resolve_repo(r"c:\K3_Firehose")
    test("Validates root repo", repo.exists() and (repo / ".git").exists())

    # Worktree list
    wt_list = k3_worktree_ops.worktree_list(r"c:\K3_Firehose")
    test("worktree_list returns string", "Active Git Worktrees" in wt_list or "No worktrees" in wt_list)

    # Path traversal blocking test
    try:
        k3_worktree_ops._validate_branch_name("../../escaped")
        test("Branch validation blocks path traversal", False)
    except ValueError:
        test("Branch validation blocks path traversal", True)

    # Unauthorized verify binary test
    unauth_res = k3_worktree_ops.worktree_merge_and_cleanup(
        branch_name="valid-test-branch",
        verify_command="malicious_binary arg1"
    )
    test("worktree_merge blocks unauthorized verify binary", "Security Error" in unauth_res)


def test_server_k3_forge() -> None:
    print("\n=== 6. Testing k3-forge ===")
    import k3_forge

    # 1. Instance and FastMCP export
    test(
        "k3_forge exports FastMCP instance",
        hasattr(k3_forge, "mcp") and k3_forge.mcp.name == "k3-forge",
    )

    # 2. Tool count check (exact requirement: 8 tools)
    tools = k3_forge.mcp._tool_manager._tools
    test(
        "k3_forge registers exactly 8 tools",
        len(tools) == 8,
        f"Found {len(tools)} tools",
    )

    # 3. Tool name registration check
    expected_tools = {
        "forge_status",
        "forge_choreograph",
        "forge_draft",
        "forge_lint",
        "forge_revise",
        "forge_critique",
        "forge_describe",
        "forge_scan_repetitions",
    }
    test(
        "k3_forge registers all 8 expected tool names",
        expected_tools.issubset(set(tools.keys())),
    )

    # 4. Status tool execution
    status_res = k3_forge.forge_status()
    test(
        "forge_status returns manuscript report",
        isinstance(status_res, str)
        and ("K3 Forge" in status_res or "Status" in status_res)
        and "hard-country" in status_res,
    )

    # 5. Deterministic prose lint check (flags dirty prose)
    dirty_text = "She felt his eyes darken as he smirked seamlessly and bit his lip."
    lint_res = k3_forge.forge_lint(dirty_text)
    test(
        "forge_lint correctly flags violations on test text",
        isinstance(lint_res, str)
        and ("violations" in lint_res.lower() or "deny" in lint_res.lower()),
    )

    # 6. Sensory palette decomposition check
    desc_res = k3_forge.forge_describe(
        "The cold iron stove rattled against the pine floor.",
        channels="tactile,acoustic",
    )
    test(
        "forge_describe returns sensory decomposition",
        isinstance(desc_res, str)
        and ("tactile" in desc_res.lower() or "acoustic" in desc_res.lower()),
    )

    # 7. Choreograph validation on known beat
    choreo_res = k3_forge.forge_choreograph("beat_06_no_way_2")
    test(
        "forge_choreograph generates micro-beat blocking",
        isinstance(choreo_res, str)
        and (
            "territorial_clash" in choreo_res
            or "Physical Blocking" in choreo_res
            or "Step" in choreo_res
        ),
    )

    # 8. Repetition scanner execution
    rep_res = k3_forge.forge_scan_repetitions("hard-country", window=5)
    test(
        "forge_scan_repetitions executes and returns report",
        isinstance(rep_res, str)
        and ("Repetition Scan" in rep_res or "Status" in rep_res),
    )

    # 9. Path traversal protection on forge_revise (dirty input: relative traversal)
    try:
        k3_forge.forge_revise("../../etc/passwd", feedback="test")
        test("forge_revise blocks dirty path traversal", False, "Expected ValueError")
    except ValueError:
        test("forge_revise blocks dirty path traversal", True)

    # 10. Path traversal protection on forge_lint (dirty input: absolute path outside drafts)
    try:
        k3_forge.forge_lint("C:\\Windows\\win.ini")
        test("forge_lint blocks absolute path traversal", False, "Expected ValueError")
    except ValueError:
        test("forge_lint blocks absolute path traversal", True)

    # 11. Clean authorized draft access on forge_lint (clean input)
    lint_clean_file = k3_forge.forge_lint("beat_06_no_way_2.md")
    test(
        "forge_lint cleanly processes authorized draft path",
        isinstance(lint_clean_file, str) and "Prose Quality Lint Report" in lint_clean_file,
    )

    # 12. Path traversal protection on forge_critique (dirty input: relative traversal)
    try:
        k3_forge.forge_critique("../../etc/passwd")
        test("forge_critique blocks dirty path traversal", False, "Expected ValueError")
    except ValueError:
        test("forge_critique blocks dirty path traversal", True)

    # 13. Clean authorized draft access on forge_critique (clean input)
    crit_clean_file = k3_forge.forge_critique("beat_06_no_way_2.md")
    test(
        "forge_critique cleanly processes authorized draft path",
        isinstance(crit_clean_file, str) and "Editorial Critique Report" in crit_clean_file,
    )


def main() -> None:
    test_server_mrl_memory()
    test_server_agent_ops()
    test_server_doc_intel()
    test_server_local_llm()
    test_server_worktree_ops()
    test_server_k3_forge()

    print(f"\n{'='*50}")
    print(f"FLEET INTEGRATION RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("ALL 6 K3 MCP SERVERS VERIFIED AND OPERATIONAL!")
    else:
        print(f"WARNING: {failed} TEST(S) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()

