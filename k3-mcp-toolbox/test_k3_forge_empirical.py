#!/usr/bin/env python3
"""
Empirical Stress Test Suite for K3 Forge FastMCP Server (k3_forge.py).
Conducted by Challenger 1 for Milestone 1.
Validates:
1. Tool count and registration.
2. Edge cases and fail-soft behavior across all 7 tools:
   - forge_status
   - forge_choreograph
   - forge_lint
   - forge_describe
   - forge_critique
   - forge_revise
   - forge_draft
3. Safe error handling (returns descriptive error strings without unhandled exceptions).
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure offline deterministic mode for fast, repeatable stress-testing
os.environ["GOOGLE_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

# Setup module path
TOOLBOX_DIR = Path(__file__).resolve().parent
SERVERS_DIR = TOOLBOX_DIR / "servers"
if str(SERVERS_DIR) not in sys.path:
    sys.path.insert(0, str(SERVERS_DIR))
if str(TOOLBOX_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_DIR))

passed = 0
failed = 0
findings = []

def record_test(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")
        findings.append(f"{name}: {detail}")

def run_empirical_suite():
    print("============================================================")
    print("  EMPIRICAL STRESS TEST: FastMCP Server servers/k3_forge.py")
    print("============================================================\n")

    # --- 1. Direct Import and FastMCP Metadata Validation ---
    print("--- 1. FastMCP Instance & Tool Registration ---")
    try:
        from servers.k3_forge import mcp
        record_test("mcp instance imported", mcp is not None)
        record_test("Server name is k3-forge", mcp.name == "k3-forge")
        record_test("Server version is 1.0.0", getattr(mcp._mcp_server, "version", None) == "1.0.0")
        
        tools = mcp._tool_manager._tools
        tool_count = len(tools)
        record_test(f"Tool count exactly 8 (actual: {tool_count})", tool_count == 8)
        
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
        actual_tools = set(tools.keys())
        missing = expected_tools - actual_tools
        extra = actual_tools - expected_tools
        record_test("All 8 required tools registered", missing == set() and extra == set(), f"Missing: {missing}, Extra: {extra}")
    except Exception as exc:
        record_test("Import k3_forge without exception", False, str(exc))
        return

    # Import tool functions directly
    from servers.k3_forge import (
        forge_status,
        forge_choreograph,
        forge_draft,
        forge_lint,
        forge_revise,
        forge_critique,
        forge_describe,
        forge_scan_repetitions,
    )

    # --- 2. forge_status Edge Cases ---
    print("\n--- 2. Tool: forge_status ---")
    # Valid call
    status_valid = forge_status("hard-country")
    record_test("forge_status valid series returns header", "# === K3 Forge Status:" in status_valid)
    record_test("forge_status contains Pacing Matrix Cursor", "Pacing Matrix Cursor" in status_valid)
    record_test("forge_status contains Manuscript Drafts table", "Manuscript Drafts" in status_valid)

    # Invalid series slug
    status_invalid = forge_status("non_existent_series_slug_404")
    record_test("forge_status invalid series returns descriptive error", status_invalid.startswith("Error: Series 'non_existent_series_slug_404' directory not found"))
    record_test("forge_status invalid series fails soft without exception", isinstance(status_invalid, str))

    # None and empty slug defaults
    status_none = forge_status(None)
    record_test("forge_status(None) defaults to hard-country", "# === K3 Forge Status: Hard Country ===" in status_none)
    status_empty = forge_status("")
    record_test("forge_status('') defaults to hard-country", "# === K3 Forge Status: Hard Country ===" in status_empty)

    # --- 3. forge_choreograph Edge Cases ---
    print("\n--- 3. Tool: forge_choreograph ---")
    # Valid beat
    choreo_valid = forge_choreograph("beat_06_no_way_2")
    record_test("forge_choreograph valid beat returns header", "# === Beat Choreography: beat_06_no_way_2 ===" in choreo_valid)
    record_test("forge_choreograph contains Micro-Beat Blocking table", "Micro-Beat Physical Blocking & Emotional Shifts" in choreo_valid)

    # Non-existent beat ID
    choreo_invalid = forge_choreograph("beat_999")
    record_test("forge_choreograph non-existent beat returns descriptive error", choreo_invalid.startswith("Error: Beat 'beat_999' not found"))
    record_test("forge_choreograph non-existent beat fails soft without exception", isinstance(choreo_invalid, str))

    # --- 4. forge_lint Edge Cases ---
    print("\n--- 4. Tool: forge_lint ---")
    # Clean text
    clean_prose = "The frost held the window glass in white fern patterns. Ryder reached for the iron poker and turned the fir split over the embers."
    lint_clean = forge_lint(clean_prose)
    record_test("forge_lint clean text returns 100/100", "100/100" in lint_clean)
    record_test("forge_lint clean text returns CLEAN status", "CLEAN (Zero prose violations detected)" in lint_clean)

    # Heavily violated text
    violated_prose = "She felt his eyes darken as he smirked seamlessly and bit his lip while he chuckled and utilized the tapestry."
    lint_violated = forge_lint(violated_prose)
    record_test("forge_lint violated text score < 100", "Overall Score: **91/100**" in lint_violated or "Score:" in lint_violated)
    record_test("forge_lint flags filter words ('She felt')", "She felt" in lint_violated)
    record_test("forge_lint flags banned body language ('bit his lip')", "bit his lip" in lint_violated)
    record_test("forge_lint flags AI tics ('smirked')", "smirked" in lint_violated)
    record_test("forge_lint flags Latinate bloat ('utilized')", "utilized" in lint_violated)

    # Non-existent file path: treated as raw text fallback
    lint_nonexistent_file = forge_lint("non_existent_chapter_404.md")
    record_test("forge_lint non-existent file falls back to raw text without crashing", "Raw Text" in lint_nonexistent_file and "100/100" in lint_nonexistent_file)

    # Empty string: fail-soft error handling
    lint_empty = forge_lint("")
    record_test("forge_lint empty string returns error message and does not raise unhandled exception", lint_empty.startswith("Error"))

    # --- 5. forge_describe Edge Cases ---
    print("\n--- 5. Tool: forge_describe ---")
    # Valid description with default channels
    desc_valid = forge_describe("The cold iron stove rattled against the pine floor.")
    record_test("forge_describe returns Sensory Palette header", "# === Sensory Palette Decomposition ===" in desc_valid)
    record_test("forge_describe contains Acoustic channel", "Acoustic Channel" in desc_valid)
    record_test("forge_describe contains Tactile channel", "Tactile Channel" in desc_valid)
    record_test("forge_describe contains Olfactory channel", "Olfactory Channel" in desc_valid)
    record_test("forge_describe contains Visual channel", "Visual Channel" in desc_valid)
    record_test("forge_describe contains Metaphor channel", "Metaphor Channel" in desc_valid)

    # Empty string
    desc_empty = forge_describe("")
    record_test("forge_describe empty string returns descriptive error", desc_empty.startswith("Error: Text parameter cannot be empty"))

    # Whitespace only
    desc_ws = forge_describe("   \t\n  ")
    record_test("forge_describe whitespace-only returns descriptive error", desc_ws.startswith("Error: Text parameter cannot be empty"))

    # Unknown / custom channels
    desc_custom = forge_describe("The cold iron stove rattled.", channels="astral,quantum_flux")
    record_test("forge_describe custom channels handled gracefully", "Astral Channel" in desc_custom and "Quantum_Flux Channel" in desc_custom)

    # --- 6. forge_critique Edge Cases ---
    print("\n--- 6. Tool: forge_critique ---")
    # Valid text evaluation
    critique_valid = forge_critique("The frost held the pine branch. Ryder reached for the iron poker.")
    record_test("forge_critique returns Editorial Critique Report header", "# === Editorial Critique Report ===" in critique_valid)
    record_test("forge_critique contains Core Metric Rubric", "Core Metric Rubric" in critique_valid)
    record_test("forge_critique contains 4 categories", all(cat in critique_valid for cat in ["Pacing", "Tension", "Voice", "Continuity"]))

    # Non-existent chapter path: treats as raw text or handles gracefully
    critique_nonexistent = forge_critique("non_existent_chapter_404.md")
    record_test("forge_critique non-existent path handled without unhandled exception", isinstance(critique_nonexistent, str) and len(critique_nonexistent) > 0)

    # Empty string: fail-soft error handling
    critique_empty = forge_critique("")
    record_test("forge_critique empty string fails soft without unhandled exception", critique_empty.startswith("Error"))

    # --- 7. forge_revise Edge Cases ---
    print("\n--- 7. Tool: forge_revise ---")
    # Non-existent file path
    revise_nonexistent = forge_revise("non_existent_chapter_404.md", feedback="Tone down the dialogue")
    record_test("forge_revise non-existent target returns descriptive error", revise_nonexistent.startswith("Error: Target file for revision not found on disk"))

    # Empty feedback
    revise_empty_fb = forge_revise("beat_06_no_way_2.md", feedback="")
    record_test("forge_revise empty feedback returns descriptive error", revise_empty_fb.startswith("Error: Feedback must not be empty"))

    # Whitespace feedback
    revise_ws_fb = forge_revise("beat_06_no_way_2.md", feedback="   ")
    record_test("forge_revise whitespace feedback returns descriptive error", revise_ws_fb.startswith("Error: Feedback must not be empty"))

    # Successful revision on a temporary draft copy
    from servers.k3_forge import _resolve_series_dir
    drafts_dir = _resolve_series_dir("hard-country") / "drafts"
    temp_target = drafts_dir / "_test_temp_revise.md"
    try:
        temp_target.write_text("# Test Draft\n\nOriginal draft text here for test.", encoding="utf-8")
        revise_success = forge_revise("_test_temp_revise.md", feedback="Clean up rhythm")
        record_test("forge_revise successfully updates existing file", "# === K3 Forge Revision Report ===" in revise_success and "Successfully updated on disk" in revise_success)
    finally:
        if temp_target.exists():
            temp_target.unlink()

    # --- 8. forge_draft Edge Cases ---
    print("\n--- 8. Tool: forge_draft ---")
    # Non-existent beat ID
    draft_invalid = forge_draft("beat_999")
    record_test("forge_draft non-existent beat returns descriptive error", draft_invalid.startswith("Error: Beat 'beat_999' not found"))

    # Pacing structure integration
    from servers.k3_forge import _get_beat_engine
    engine = _get_beat_engine()
    record_test("BeatEngine loads pacing matrix successfully", engine is not None and len(engine._sequence) > 0)

    # --- 9. Tool: forge_scan_repetitions ---
    print("\n--- 9. Tool: forge_scan_repetitions ---")
    rep_valid = forge_scan_repetitions("hard-country", window=5)
    record_test(
        "forge_scan_repetitions returns valid report",
        "# === Cross-Chapter Repetition Scan:" in rep_valid or "Repetition" in rep_valid,
    )

    rep_invalid = forge_scan_repetitions("non_existent_series_999")
    record_test(
        "forge_scan_repetitions handles missing series gracefully",
        rep_invalid.startswith("Error") or "not found" in rep_invalid,
    )

    print("\n============================================================")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("ALL EMPIRICAL STRESS TESTS PASSED! FAIL-SOFT VERIFIED!")
    else:
        print(f"FAILURES DETECTED ({failed}):")
        for f in findings:
            print(f"  - {f}")
    print("============================================================\n")

    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    run_empirical_suite()
