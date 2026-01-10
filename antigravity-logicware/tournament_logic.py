from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class AuditorSignal(BaseModel):
    """
    Structured output for the Reflection Node (The Auditor).
    Acts as a logic gate for the Three-Brain Protocol.
    """

    status: Literal["PASS", "FAIL", "pass", "fail"] = Field(
        ..., description="Binary gate decision. FAIL triggers a regeneration loop."
    )
    risk_vector: List[str] = Field(
        default_factory=list,
        description="Specific constraints violated (e.g., 'Fiduciary Risk', 'Safety', 'Time Cost').",
    )
    mutation_hint: str = Field(
        ...,
        description="Directive for the Creator on how to shift the next generation to escape the local minimum.",
    )

    @field_validator("status")
    def normalize_status(cls, v):
        return v.upper()


class TemperatureDecoupler:
    """
    Manages entropy injection for the Generation Node.
    """

    def __init__(
        self, base_temp: float = 0.4, step: float = 0.2, max_temp: float = 1.0
    ):
        self.current_temp = base_temp
        self.step = step
        self.max_temp = max_temp
        self.retry_count = 0

    def adapt(self) -> float:
        """Increases temperature based on retry count."""
        self.retry_count += 1
        new_temp = self.current_temp + (self.retry_count * self.step)
        return min(new_temp, self.max_temp)

    def reset(self, base_temp: float = 0.4):
        self.current_temp = base_temp
        self.retry_count = 0


def infinite_loop_guard(
    history: List[AuditorSignal], current_signal: AuditorSignal
) -> bool:
    """
    Detects stagnation by checking if the Creator is receiving the EXACT same feedback repeatedly.
    """
    if not history:
        return False

    last_signal = history[-1]

    # If the risk vector AND the hint are identical, the Creator isn't listening (or is stuck).
    if (
        current_signal.status == "FAIL"
        and last_signal.status == "FAIL"
        and current_signal.risk_vector == last_signal.risk_vector
        and current_signal.mutation_hint == last_signal.mutation_hint
    ):
        return True

    return False
