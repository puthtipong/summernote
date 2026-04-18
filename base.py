"""
Shared dataclasses for the evolutionary LLM red-team loop.

These types are the wire format between the three stages:
  - mutator.LmMutator    — Propose
  - scorer.Scorer        — Score
  - controller.Controller — Select + Update (MAP-Elites)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Usage / cost tracking
# ---------------------------------------------------------------------------


@dataclass
class UsageAccumulator:
    """Collects token usage across all API calls in a run.

    Pass one instance to both LmMutator and Scorer; it accumulates in-place.
    Read final totals from RunResult.usage.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0   # subset of completion_tokens, gpt-5.x only
    api_calls: int = 0

    def add(self, resp_usage: Any) -> None:
        """Accept an openai.types.CompletionUsage (or None) and accumulate."""
        if resp_usage is None:
            return
        self.prompt_tokens += getattr(resp_usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(resp_usage, "completion_tokens", 0) or 0
        details = getattr(resp_usage, "completion_tokens_details", None)
        if details is not None:
            self.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0
        self.api_calls += 1

    def summary(self) -> str:
        return (
            f"api_calls={self.api_calls} "
            f"prompt={self.prompt_tokens} "
            f"completion={self.completion_tokens} "
            f"(reasoning={self.reasoning_tokens})"
        )


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetResponse:
    """What the target system returned after the trigger was injected.

    For agentic targets, tool_calls holds the list of invocations observed
    (e.g. [{"name": "send_email", "args": {"to": "..."}}]).
    For plain-chat targets, tool_calls is empty and text carries the reply.
    raw is the original SDK response object — useful for custom oracles.
    """
    text: str
    tool_calls: tuple[dict, ...] = field(default_factory=tuple)
    raw: Any = field(default=None, compare=False, repr=False)

    def short(self, n: int = 600) -> str:
        body = self.text if len(self.text) <= n else self.text[:n] + "…(truncated)"
        if self.tool_calls:
            body += f"\n[tool_calls={list(self.tool_calls)}]"
        return body

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tool_calls": list(self.tool_calls),
            # raw is intentionally omitted — not serialisable in general
        }

    @classmethod
    def from_dict(cls, d: dict) -> TargetResponse:
        return cls(
            text=d["text"],
            tool_calls=tuple(d.get("tool_calls", [])),
        )


@dataclass
class ScoreResult:
    """Output of scoring a single trigger."""
    score: float                    # critic 1-10 (gradient signal)
    feedback: str                   # short qualitative explanation
    succeeded: bool                 # hard oracle — the ground truth
    target_response: TargetResponse


@dataclass
class Candidate:
    """A scored trigger living in the evolutionary pool."""
    text: str
    score: float
    succeeded: bool
    target_response: TargetResponse
    feedback: str
    iteration: int
    source: str                     # "base" | "seed:<id>" | "llm:<strategy>"

    def summary(self, n: int = 400) -> str:
        head = f"[iter={self.iteration} src={self.source} score={self.score:.1f}"
        head += " ORACLE=✓]" if self.succeeded else "]"
        body = self.text if len(self.text) <= n else self.text[:n] + "…"
        return f"{head}\n{body}"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": self.score,
            "succeeded": self.succeeded,
            "target_response": self.target_response.to_dict(),
            "feedback": self.feedback,
            "iteration": self.iteration,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Candidate:
        return cls(
            text=d["text"],
            score=float(d["score"]),
            succeeded=bool(d["succeeded"]),
            target_response=TargetResponse.from_dict(d["target_response"]),
            feedback=d["feedback"],
            iteration=int(d["iteration"]),
            source=d["source"],
        )
