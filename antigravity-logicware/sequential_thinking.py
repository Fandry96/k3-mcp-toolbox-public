from typing import Optional, List, Dict, Any
import json


class SequentialThinking:
    """
    Protocol 310: Sequential Thinking Meta-Tool
    (K3 Firehose Python Implementation)

    Based on the Model Context Protocol Community Specification.
    A tool for dynamic and reflective problem-solving through thoughts.
    """

    def __init__(self):
        self.history = []

    def execute(
        self,
        thought: str,
        next_thought_needed: bool,
        thought_number: int,
        total_thoughts: int,
        is_revision: bool = False,
        revises_thought: Optional[int] = None,
        branch_from_thought: Optional[int] = None,
        branch_id: Optional[str] = None,
        needs_more_thoughts: bool = False,
    ) -> str:
        """
        Records a thinking step.
        """

        step_data = {
            "thought": thought,
            "next_thought_needed": next_thought_needed,
            "thought_number": thought_number,
            "total_thoughts": total_thoughts,
            "is_revision": is_revision,
            "revises_thought": revises_thought,
            "branch_from_thought": branch_from_thought,
            "branch_id": branch_id,
            "needs_more_thoughts": needs_more_thoughts,
        }

        self.history.append(step_data)

        # In a real MCP server, this would just return confirmation.
        # For our local tool, we return a formatted string.

        status = "CONTINUING" if next_thought_needed else "COMPLETE"

        output = f"""
[Thinking Step {thought_number}/{total_thoughts}] ({status})
--------------------------------------------------
{thought}
--------------------------------------------------
"""
        return output

    def get_tool_definition(self):
        return {
            "name": "sequential_thinking",
            "description": "A detailed tool for dynamic and reflective problem-solving through thoughts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your current thinking step.",
                    },
                    "next_thought_needed": {
                        "type": "boolean",
                        "description": "Whether another thought step is needed.",
                    },
                    "thought_number": {
                        "type": "integer",
                        "description": "Current thought number.",
                    },
                    "total_thoughts": {
                        "type": "integer",
                        "description": "Estimated total thoughts needed.",
                    },
                    "is_revision": {
                        "type": "boolean",
                        "description": "Whether this revises previous thinking.",
                    },
                    "revises_thought": {
                        "type": "integer",
                        "description": "Which thought is being reconsidered.",
                    },
                    "branch_from_thought": {
                        "type": "integer",
                        "description": "Branching point thought number.",
                    },
                    "branch_id": {
                        "type": "string",
                        "description": "Branch identifier.",
                    },
                    "needs_more_thoughts": {
                        "type": "boolean",
                        "description": "If more thoughts are needed.",
                    },
                },
                "required": [
                    "thought",
                    "next_thought_needed",
                    "thought_number",
                    "total_thoughts",
                ],
            },
        }
