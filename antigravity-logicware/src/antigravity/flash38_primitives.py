"""
Gemini 3.8 Flash — Agentic Primitives
(Stolen from googleapis/python-genai + google-gemini/cookbook, Sept 2 2026)

Provides:
    - FlashConfig: Safe config builder that enforces 3.8 Flash constraints
    - UpdateTool: The structured update() tool declaration + handler
    - AgentLoop: Production multi-turn agent loop with ID matching
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# 1. Config Sanitizer — strips banned params before API dispatch
# ---------------------------------------------------------------------------

class ThinkingLevel(str, Enum):
    """Valid thinking levels for Gemini 3.8 Flash."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Params that trigger HTTP 400 INVALID_ARGUMENT on 3.8 Flash
_BANNED_PARAMS = frozenset({
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "frequency_penalty",
    "candidate_count",
})

# Mutually exclusive with thinking_level
_INCOMPATIBLE_PARAMS = frozenset({
    "thinking_budget",
    "reasoning_effort",
})


@dataclass
class FlashConfig:
    """
    Safe configuration builder for gemini-3.8-flash.

    Enforces:
    - thinking_level must be low/medium/high (minimal → crash)
    - Banned sampling params are rejected at construction
    - thinking_budget/reasoning_effort cannot coexist with thinking_level

    Usage:
        config = FlashConfig(thinking_level=ThinkingLevel.MEDIUM)
        api_kwargs = config.to_api_dict()
    """

    thinking_level: ThinkingLevel = ThinkingLevel.MEDIUM
    include_thoughts: bool = False
    max_output_tokens: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    response_mime_type: Optional[str] = None
    response_schema: Optional[dict] = None

    def __post_init__(self):
        # Guard: reject 'minimal' even if someone passes the raw string
        if isinstance(self.thinking_level, str):
            normalized = self.thinking_level.lower().strip()
            if normalized == "minimal":
                raise ValueError(
                    "thinking_level='minimal' is unsupported on gemini-3.8-flash. "
                    "Use 'low', 'medium', or 'high'."
                )
            self.thinking_level = ThinkingLevel(normalized)

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to google-genai GenerateContentConfig kwargs."""
        config = {
            "thinking_config": {
                "thinking_level": self.thinking_level.value.upper(),
            }
        }
        if self.include_thoughts:
            config["thinking_config"]["include_thoughts"] = True
        if self.max_output_tokens is not None:
            config["max_output_tokens"] = self.max_output_tokens
        if self.stop_sequences:
            config["stop_sequences"] = self.stop_sequences
        if self.response_mime_type:
            config["response_mime_type"] = self.response_mime_type
        if self.response_schema:
            config["response_schema"] = self.response_schema
        return config

    @staticmethod
    def sanitize_kwargs(raw_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Strip banned params from an arbitrary config dict.
        Use this when proxying through LiteLLM or other frameworks
        that may pass legacy params.

        Returns a clean copy; original is not mutated.
        """
        cleaned = {}
        violations = []
        for k, v in raw_kwargs.items():
            if k in _BANNED_PARAMS:
                violations.append(k)
            elif k in _INCOMPATIBLE_PARAMS:
                violations.append(f"{k} (incompatible with thinking_level)")
            else:
                cleaned[k] = v
        if violations:
            import warnings
            warnings.warn(
                f"Stripped banned 3.8 Flash params: {violations}",
                stacklevel=2,
            )
        return cleaned


# ---------------------------------------------------------------------------
# 2. update() Tool Declaration — structured progress reporting
# ---------------------------------------------------------------------------

# JSON Schema (for REST / MCP registration)
UPDATE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "update",
    "description": (
        "Report execution progress, plan status, and immediate next step "
        "during multi-step agent workflows. MUST be called before executing "
        "any actionable tool."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "previous_step": {
                "type": "STRING",
                "description": "Summary of what was just accomplished or observed.",
            },
            "plan": {
                "type": "STRING",
                "description": "High-level remaining roadmap or tasks.",
            },
            "next_step": {
                "type": "STRING",
                "description": "The specific immediate action intended.",
            },
            "external": {
                "type": "STRING",
                "description": "Optional notes, external references, or shared context.",
            },
        },
        "required": ["previous_step", "plan", "next_step"],
    },
}


def get_update_tool_declaration():
    """
    Returns the update() FunctionDeclaration for the google-genai SDK.

    Usage:
        from google.genai import types
        tools = [types.Tool(function_declarations=[
            get_update_tool_declaration(),
            ...your_other_tools...
        ])]
    """
    try:
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai SDK required. Install: pip install google-genai"
        )

    return types.FunctionDeclaration(
        name="update",
        description=UPDATE_TOOL_SCHEMA["description"],
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "previous_step": types.Schema(
                    type=types.Type.STRING,
                    description="Summary of what was just accomplished or observed.",
                ),
                "plan": types.Schema(
                    type=types.Type.STRING,
                    description="High-level remaining roadmap or tasks.",
                ),
                "next_step": types.Schema(
                    type=types.Type.STRING,
                    description="The specific immediate action intended.",
                ),
                "external": types.Schema(
                    type=types.Type.STRING,
                    description="Optional notes, external references, or shared context.",
                ),
            },
            required=["previous_step", "plan", "next_step"],
        ),
    )


@dataclass
class UpdateState:
    """Accumulates update() calls for session telemetry."""

    history: list[dict[str, str]] = field(default_factory=list)

    def handle(self, args: dict[str, str]) -> dict[str, str]:
        """Process an update() tool call. Returns acknowledgment payload."""
        entry = {
            "previous_step": args.get("previous_step", ""),
            "plan": args.get("plan", ""),
            "next_step": args.get("next_step", ""),
        }
        if "external" in args:
            entry["external"] = args["external"]
        self.history.append(entry)
        return {"status": "acknowledged", "turn": len(self.history)}

    @property
    def last(self) -> Optional[dict[str, str]]:
        return self.history[-1] if self.history else None


# ---------------------------------------------------------------------------
# 3. Agent Loop — production multi-turn with strict ID matching
# ---------------------------------------------------------------------------

@dataclass
class ToolDispatcher:
    """
    Dispatch table for tool execution with FunctionCall.id propagation.

    Usage:
        dispatcher = ToolDispatcher()
        dispatcher.register("read_file", my_read_file_fn)
        dispatcher.register("bash", my_bash_fn)

        # The update tool is auto-registered
        loop = AgentLoop(dispatcher=dispatcher)
    """

    _handlers: dict[str, Callable[..., dict]] = field(default_factory=dict)
    _update_state: UpdateState = field(default_factory=UpdateState)

    def register(self, name: str, handler: Callable[..., dict]):
        """Register a tool handler. Handler receives (args: dict) -> dict."""
        self._handlers[name] = handler

    def execute(self, name: str, args: dict) -> dict:
        """Execute a tool by name. Returns result dict."""
        if name == "update":
            return self._update_state.handle(args)
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        return handler(args)

    @property
    def update_history(self) -> list[dict[str, str]]:
        return self._update_state.history


def run_agent_loop(
    client,
    model: str,
    prompt: str,
    tools: list,
    dispatcher: ToolDispatcher,
    *,
    thinking_level: str = "medium",
    system_instruction: str = "",
    max_turns: int = 15,
    on_update: Optional[Callable[[dict], None]] = None,
    on_final: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Production multi-turn agent loop for Gemini 3.8 Flash.

    Handles:
    - Structured update() interception
    - Strict FunctionCall.id ↔ FunctionResponse.id matching
    - Parallel tool call batching
    - Turn limiting

    Args:
        client: google.genai.Client instance
        model: Model string (e.g., "gemini-3.8-flash")
        prompt: Initial user prompt
        tools: List of types.Tool declarations (include update tool)
        dispatcher: ToolDispatcher with registered handlers
        thinking_level: "low" | "medium" | "high"
        system_instruction: System prompt string
        max_turns: Maximum tool-call turns before force-stopping
        on_update: Optional callback fired on each update() call
        on_final: Optional callback fired with the final text response

    Returns:
        Final text response from the model.
    """
    try:
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai SDK required.")

    # Validate thinking level
    config_obj = FlashConfig(thinking_level=thinking_level)

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level=config_obj.thinking_level.value.upper()
        ),
        tools=tools,
    )
    if system_instruction:
        config.system_instruction = system_instruction

    chat = client.chats.create(model=model, config=config)
    response = chat.send_message(prompt)

    for turn in range(max_turns):
        candidate = response.candidates[0]

        # Collect all function calls (handles parallel calls)
        function_calls = [
            part.function_call
            for part in candidate.content.parts
            if part.function_call
        ]

        if not function_calls:
            # Final answer
            final_text = response.text or ""
            if on_final:
                on_final(final_text)
            return final_text

        # Execute all calls, build matched responses
        response_parts = []
        for fc in function_calls:
            result = dispatcher.execute(fc.name, fc.args or {})

            if fc.name == "update" and on_update:
                on_update(fc.args or {})

            # Strict ID matching
            response_parts.append(
                types.Part.from_function_response(
                    id=fc.id,       # CRITICAL: must match FunctionCall.id
                    name=fc.name,
                    response={"result": result},
                )
            )

        response = chat.send_message(response_parts)

    return "[Agent loop exceeded max_turns]"
