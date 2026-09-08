"""Smoke test for flash38_primitives — no google-genai SDK required."""
import sys, warnings
sys.path.insert(0, ".")

from antigravity.flash38_primitives import (
    FlashConfig, ThinkingLevel, UpdateState, ToolDispatcher, UPDATE_TOOL_SCHEMA
)

# 1. FlashConfig — happy path
c = FlashConfig(thinking_level="medium")
d = c.to_api_dict()
assert d["thinking_config"]["thinking_level"] == "MEDIUM", f"Expected MEDIUM, got {d}"
print("✅ FlashConfig(medium) →", d)

# 2. FlashConfig — minimal must crash
try:
    FlashConfig(thinking_level="minimal")
    print("❌ minimal should have raised ValueError")
    sys.exit(1)
except ValueError as e:
    print(f"✅ minimal BLOCKED: {e}")

# 3. FlashConfig — enum input
c2 = FlashConfig(thinking_level=ThinkingLevel.HIGH)
assert c2.to_api_dict()["thinking_config"]["thinking_level"] == "HIGH"
print("✅ ThinkingLevel.HIGH enum works")

# 4. Sanitizer — strips banned params
raw = {
    "temperature": 0.7,
    "top_p": 0.9,
    "presence_penalty": 0.5,
    "thinking_config": {"thinking_level": "MEDIUM"},
    "max_output_tokens": 8192,
}
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    clean = FlashConfig.sanitize_kwargs(raw)
    assert "temperature" not in clean
    assert "top_p" not in clean
    assert "presence_penalty" not in clean
    assert "thinking_config" in clean
    assert "max_output_tokens" in clean
    assert len(w) == 1  # one warning about stripped params
    print(f"✅ Sanitizer stripped 3 params, kept 2: {list(clean.keys())}")
    print(f"   Warning: {w[0].message}")

# 5. UpdateState — accumulation
us = UpdateState()
r1 = us.handle({"previous_step": "read file", "plan": "analyze", "next_step": "grep"})
assert r1["status"] == "acknowledged"
assert r1["turn"] == 1
r2 = us.handle({"previous_step": "grepped", "plan": "report", "next_step": "write"})
assert r2["turn"] == 2
assert len(us.history) == 2
assert us.last["next_step"] == "write"
print(f"✅ UpdateState: {len(us.history)} entries, last={us.last}")

# 6. ToolDispatcher — routing
td = ToolDispatcher()
td.register("echo", lambda args: {"echoed": args.get("msg", "")})

# update() auto-handled
ur = td.execute("update", {"previous_step": "init", "plan": "test", "next_step": "echo"})
assert ur["status"] == "acknowledged"

# registered tool
er = td.execute("echo", {"msg": "hello"})
assert er["echoed"] == "hello"

# unknown tool
unk = td.execute("nonexistent", {})
assert "error" in unk
print(f"✅ ToolDispatcher: update={ur}, echo={er}, unknown={unk}")

# 7. UPDATE_TOOL_SCHEMA — structure check
assert UPDATE_TOOL_SCHEMA["name"] == "update"
assert "previous_step" in UPDATE_TOOL_SCHEMA["parameters"]["properties"]
assert "plan" in UPDATE_TOOL_SCHEMA["parameters"]["properties"]
assert "next_step" in UPDATE_TOOL_SCHEMA["parameters"]["properties"]
print(f"✅ UPDATE_TOOL_SCHEMA valid, {len(UPDATE_TOOL_SCHEMA['parameters']['properties'])} properties")

print("\n🏁 ALL TESTS PASSED")
