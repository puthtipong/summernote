"""
Controller — MAP-Elites evolutionary search for adversarial triggers.

This is the Select + Update half of the PSSU loop, plus the outer
orchestration. Maintains a 2D grid of elite candidates keyed by
(length_bucket, structure_bucket); one elite per cell. Each iteration:

  1. Sample a batch of past candidates (mix of elites + random-recent).
  2. LmMutator proposes N new triggers (one API call).
  3. Score all N concurrently (target + critic per trigger).
  4. Update the grid: a candidate is admitted if its cell is empty or
     its score beats the cell's current elite.
  5. Check termination: oracle success, max_iterations, or budget exhausted.

Phase 0 ("seed") scores the base prompt and each provided deterministic
transform variant before the first mutator call — free early diversity.

Design notes on return_exceptions
----------------------------------
asyncio.gather is called with return_exceptions=True. If a single scorer
call fails (transient rate-limit, network blip), we log the error and skip
that candidate rather than aborting the entire batch. A partial batch is
still valuable — we keep the successful candidates and update the grid.
If ALL scoring tasks in a batch fail, that signals a systemic problem
(auth failure, quota exhausted) and we raise rather than silently continuing
with no new data.

State persistence
-----------------
Pass checkpoint_path to RunConfig to enable automatic checkpointing. The
state is written after every checkpoint_every iterations and on completion.
Call controller.load_state(path) before run() to resume a previous run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

from base import Candidate, UsageAccumulator
from mutator import LmMutator
from scorer import Scorer

logger = logging.getLogger(__name__)

SeedTransform = Callable[[str], Awaitable[str]]


# ---------------------------------------------------------------------------
# Configuration & result
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    max_iterations: int = 50
    batch_size: int = 8             # expected mutator proposals per iter (used for budget pre-check)
    sample_size: int = 8            # past candidates fed to the mutator per iter
    elite_bias: float = 0.5         # fraction of sample drawn from grid elites
    stop_on_success: bool = True    # halt on first oracle win
    max_api_calls: int = 2000       # hard safety cap
    checkpoint_path: Optional[str] = None   # write state here; None = no checkpointing
    checkpoint_every: int = 5       # iterations between checkpoints
    rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.sample_size < 0:
            raise ValueError("sample_size must be >= 0")
        if not (0.0 <= self.elite_bias <= 1.0):
            raise ValueError("elite_bias must be in [0.0, 1.0]")
        if self.max_api_calls < 10:
            raise ValueError("max_api_calls must be >= 10")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")


@dataclass
class RunResult:
    succeeded: bool
    best: Optional[Candidate]
    all_successes: list[Candidate] = field(default_factory=list)
    iterations: int = 0
    elites: int = 0
    api_calls: int = 0
    total_scored: int = 0
    usage: UsageAccumulator = field(default_factory=UsageAccumulator)


# ---------------------------------------------------------------------------
# MAP-Elites grid
#
# Two axes chosen to force structural diversity without embeddings:
#   LENGTH:    5 buckets (<50, <200, <500, <1500, ≥1500 chars)
#   STRUCTURE: 6 buckets covering dominant signatures our mutator produces
#
# Grid capacity: 5 × 6 = 30 cells.
# ---------------------------------------------------------------------------

_LENGTH_EDGES = (50, 200, 500, 1500)


def _length_bucket(text: str) -> int:
    n = len(text)
    for i, edge in enumerate(_LENGTH_EDGES):
        if n < edge:
            return i
    return len(_LENGTH_EDGES)  # bucket 4: ≥1500 chars


def _structure_bucket(text: str) -> int:
    # Priority-ordered: first matching pattern wins.
    if any(m in text for m in (
        "<|", "[START", "[/END", "####################",
        "[System Status", "<eos>", "<|endoftext|>",
    )):
        return 5  # control / fake-token markers
    lower = text.lower()
    if any(r in lower for r in ("user:", "assistant:", "system:")):
        return 1  # chat role markers
    stripped = text.lstrip()
    if "```" in text or stripped.startswith(("# ", "## ", "### ")):
        return 2  # markdown / code fence
    if any(ord(c) > 127 for c in text):
        return 3  # non-ASCII (unicode smuggling, confusables, emoji)
    if any(kw in text for kw in (
        "1.", "2.", "Step 1", "MUST", "FORBIDDEN", "REQUIRED", "DISABLED",
    )):
        return 4  # structured / imperative list
    return 0  # plain prose


def _cell(text: str) -> tuple[int, int]:
    return (_length_bucket(text), _structure_bucket(text))


class MapElitesGrid:
    """One elite per (length, structure) cell; higher score displaces lower."""

    def __init__(self) -> None:
        self._cells: dict[tuple[int, int], Candidate] = {}

    def add(self, cand: Candidate) -> bool:
        """True iff candidate was admitted (new cell OR higher than existing elite)."""
        cell = _cell(cand.text)
        existing = self._cells.get(cell)
        if existing is None or cand.score > existing.score:
            self._cells[cell] = cand
            return True
        return False

    def elites(self) -> list[Candidate]:
        return list(self._cells.values())

    def size(self) -> int:
        return len(self._cells)

    def best(self) -> Optional[Candidate]:
        if not self._cells:
            return None
        return max(self._cells.values(), key=lambda c: c.score)

    def to_dict(self) -> dict:
        return {
            f"{k[0]},{k[1]}": v.to_dict()
            for k, v in self._cells.items()
        }

    def load_dict(self, d: dict) -> None:
        self._cells = {}
        for key, val in d.items():
            parts = key.split(",")
            if len(parts) != 2:
                raise ValueError(f"Invalid grid key in checkpoint: {key!r}")
            cell = (int(parts[0]), int(parts[1]))
            self._cells[cell] = Candidate.from_dict(val)


# ---------------------------------------------------------------------------
# Helper: build seed_transforms from a transforms.py-style registry
# ---------------------------------------------------------------------------


def transforms_from_registry(
    registry: dict, include_llm: bool = False
) -> list[tuple[str, SeedTransform]]:
    """
    Convert a transforms.TRANSFORMS registry into seed_transforms pairs.

    registry : dict[str, TransformDef]  — must expose .apply_async and .requires_llm
    include_llm : if False (default), skip transforms that need an LLM client.

    Example:
        import sys; sys.path.insert(0, "../llm-fuzzer")
        from transforms import TRANSFORMS
        seeds = transforms_from_registry(TRANSFORMS)
    """
    pairs: list[tuple[str, SeedTransform]] = []
    for tid, td in registry.items():
        if getattr(td, "requires_llm", False) and not include_llm:
            continue

        def _make_fn(td_ref):
            async def _call(prompt: str) -> str:
                return await td_ref.apply_async(prompt, None, "")
            return _call

        pairs.append((tid, _make_fn(td)))
    return pairs


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------


class Controller:
    def __init__(
        self,
        mutator: LmMutator,
        scorer: Scorer,
        base_prompt: str,
        seed_transforms: Optional[list[tuple[str, SeedTransform]]] = None,
        config: Optional[RunConfig] = None,
    ) -> None:
        if not isinstance(base_prompt, str) or not base_prompt.strip():
            raise ValueError("base_prompt must be a non-empty string")
        if seed_transforms is not None:
            if not isinstance(seed_transforms, list):
                raise TypeError("seed_transforms must be a list of (name, fn) pairs")
            for item in seed_transforms:
                if (
                    not isinstance(item, tuple)
                    or len(item) != 2
                    or not isinstance(item[0], str)
                    or not callable(item[1])
                ):
                    raise TypeError(
                        "each seed_transforms entry must be (name: str, fn: async callable)"
                    )

        self._mutator = mutator
        self._scorer = scorer
        self._base_prompt = base_prompt
        self._seed_transforms = list(seed_transforms) if seed_transforms else []
        self._cfg = config or RunConfig()
        self._rng = random.Random(self._cfg.rng_seed)

        # Warn if batch_size and mutator.n_candidates disagree — the pre-flight
        # budget check uses batch_size; the actual scoring uses len(proposals).
        # A mismatch doesn't break anything but can lead to confusing log output.
        if self._cfg.batch_size != mutator.n_candidates:
            logger.warning(
                "RunConfig.batch_size=%d but LmMutator.n_candidates=%d. "
                "These should match. The mutator will propose %d candidates; "
                "the budget pre-check uses %d.",
                self._cfg.batch_size, mutator.n_candidates,
                mutator.n_candidates, self._cfg.batch_size,
            )

        self._grid = MapElitesGrid()
        self._history: list[Candidate] = []
        self._successes: list[Candidate] = []
        self._api_calls = 0
        self._iter = 0

    # ---------- bookkeeping -------------------------------------------------

    def _budget_left(self) -> int:
        return self._cfg.max_api_calls - self._api_calls

    def _early_stop(self) -> bool:
        return self._cfg.stop_on_success and bool(self._successes)

    # ---------- scoring -------------------------------------------------

    async def _score_and_absorb(
        self, text: str, source: str, iter_num: int
    ) -> Candidate:
        result = await self._scorer.score(text)
        self._api_calls += 2  # target call + critic call

        cand = Candidate(
            text=text,
            score=result.score,
            succeeded=result.succeeded,
            target_response=result.target_response,
            feedback=result.feedback,
            iteration=iter_num,
            source=source,
        )
        self._history.append(cand)
        if cand.succeeded:
            self._successes.append(cand)
        admitted = self._grid.add(cand)

        logger.info(
            "    score=%.1f oracle=%s admitted=%s src=%s | %s",
            cand.score, cand.succeeded, admitted, source,
            cand.feedback[:120],
        )
        return cand

    # ---------- history sampling --------------------------------------------

    def _sample_history(self, n: int) -> list[Candidate]:
        """Up to n past candidates: top elites + random non-elite from recent history."""
        if n <= 0 or not self._history:
            return []
        elites = self._grid.elites()
        n_elite = min(len(elites), max(1, int(round(n * self._cfg.elite_bias))))
        n_rand = n - n_elite

        top_elites = sorted(elites, key=lambda c: -c.score)[:n_elite]
        elite_ids = {id(c) for c in top_elites}

        recent_window = max(20, n_rand * 3) if n_rand > 0 else 0
        recent = self._history[-recent_window:] if recent_window else []
        non_elite = [c for c in recent if id(c) not in elite_ids]
        rand_pick: list[Candidate] = []
        if n_rand > 0 and non_elite:
            rand_pick = self._rng.sample(non_elite, min(n_rand, len(non_elite)))
        return top_elites + rand_pick

    # ---------- phases ------------------------------------------------------

    async def _seed(self) -> None:
        logger.info(
            "=== seed phase: base_prompt + %d transforms ===",
            len(self._seed_transforms),
        )
        await self._score_and_absorb(self._base_prompt, "base", iter_num=0)
        if self._early_stop():
            return

        for name, fn in self._seed_transforms:
            if self._budget_left() <= 2:
                logger.warning("seed: budget exhausted; stopping seed phase early")
                return
            try:
                mutated = await fn(self._base_prompt)
            except Exception as e:
                logger.warning("seed: transform %r raised %s; skipping", name, e)
                continue
            if not isinstance(mutated, str) or not mutated:
                logger.warning(
                    "seed: transform %r returned non-string or empty; skipping", name
                )
                continue
            if mutated == self._base_prompt:
                logger.debug("seed: transform %r is a no-op; skipping", name)
                continue
            await self._score_and_absorb(mutated, f"seed:{name}", iter_num=0)
            if self._early_stop():
                return

    async def _evolve_step(self) -> None:
        sampled = self._sample_history(self._cfg.sample_size)
        logger.info(
            "=== iter %d | grid=%d/30 | history=%d | api=%d/%d ===",
            self._iter, self._grid.size(), len(self._history),
            self._api_calls, self._cfg.max_api_calls,
        )

        proposals = await self._mutator.propose(sampled)
        self._api_calls += 1  # one mutator call

        logger.info("  mutator returned %d proposals", len(proposals))

        # Budget check against actual proposal count (not config.batch_size).
        per_score_cost = 2
        needed = len(proposals) * per_score_cost
        if self._budget_left() < needed:
            logger.warning(
                "iter %d: need %d api units to score all proposals but only %d left; "
                "skipping this iteration",
                self._iter, needed, self._budget_left(),
            )
            return

        # Score all proposals concurrently.
        # return_exceptions=True: a single transient failure (rate limit, network)
        # does not abort the whole batch. We absorb successful candidates and log
        # failures. If ALL fail, that is systemic and we raise.
        tasks = [
            self._score_and_absorb(p.trigger, f"llm:{p.strategy}", self._iter)
            for p in proposals
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        failed: list[BaseException] = []
        for r in results:
            if isinstance(r, BaseException):
                failed.append(r)
                logger.error(
                    "iter %d: scoring task failed: %s", self._iter, r,
                    exc_info=r,
                )

        if len(failed) == len(results):
            raise RuntimeError(
                f"iter {self._iter}: ALL {len(results)} scoring tasks failed. "
                "This indicates a systemic issue (auth, quota, network). "
                f"Last error: {failed[-1]!r}"
            )
        if failed:
            logger.warning(
                "iter %d: %d/%d scoring tasks failed; pool updated with %d candidates",
                self._iter, len(failed), len(results), len(results) - len(failed),
            )

    # ---------- state persistence -------------------------------------------

    def save_state(self, path: str) -> None:
        """Serialise run state to a JSON file (resumable)."""
        state = {
            "iter": self._iter,
            "api_calls": self._api_calls,
            "history": [c.to_dict() for c in self._history],
            "grid": self._grid.to_dict(),
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        Path(tmp).replace(path)  # atomic rename
        logger.info("checkpoint saved → %s (history=%d)", path, len(self._history))

    def load_state(self, path: str) -> None:
        """Restore run state from a JSON checkpoint produced by save_state()."""
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        self._iter = int(state["iter"])
        self._api_calls = int(state["api_calls"])
        self._history = [Candidate.from_dict(d) for d in state["history"]]
        self._successes = [c for c in self._history if c.succeeded]
        self._grid.load_dict(state["grid"])
        logger.info(
            "checkpoint loaded ← %s (iter=%d, history=%d, elites=%d)",
            path, self._iter, len(self._history), self._grid.size(),
        )

    def _maybe_checkpoint(self) -> None:
        path = self._cfg.checkpoint_path
        if path and self._iter % self._cfg.checkpoint_every == 0:
            try:
                self.save_state(path)
            except Exception as e:
                # Checkpointing failure must not kill the run.
                logger.error("checkpoint failed: %s", e)

    # ---------- entry point -------------------------------------------------

    async def run(self) -> RunResult:
        logger.info(
            "Controller.run: prompt=%r (%d chars) seed_transforms=%d "
            "max_iter=%d batch=%d budget=%d",
            self._base_prompt[:80], len(self._base_prompt),
            len(self._seed_transforms), self._cfg.max_iterations,
            self._cfg.batch_size, self._cfg.max_api_calls,
        )

        # Phase 0: seed.
        await self._seed()
        if self._early_stop():
            logger.info("✓ oracle satisfied during seed phase")
            return self._finalize()

        # Phase 1..N: evolutionary loop.
        for i in range(1, self._cfg.max_iterations + 1):
            self._iter = i
            min_needed = 1 + self._cfg.batch_size * 2  # 1 mutator + N scorers
            if self._budget_left() < min_needed:
                logger.warning(
                    "budget exhausted before iter %d (need %d, have %d); stopping",
                    i, min_needed, self._budget_left(),
                )
                break

            await self._evolve_step()
            self._maybe_checkpoint()

            if self._early_stop():
                logger.info("✓ oracle satisfied at iter %d", i)
                break

        return self._finalize()

    def _finalize(self) -> RunResult:
        if self._cfg.checkpoint_path:
            try:
                self.save_state(self._cfg.checkpoint_path)
            except Exception as e:
                logger.error("final checkpoint failed: %s", e)

        best = self._grid.best()
        usage = getattr(self._mutator, "_usage", None) or UsageAccumulator()
        logger.info(
            "run complete: iters=%d elites=%d successes=%d scored=%d api=%d "
            "best_score=%s | %s",
            self._iter, self._grid.size(), len(self._successes),
            len(self._history), self._api_calls,
            f"{best.score:.1f}" if best else "n/a",
            usage.summary(),
        )
        return RunResult(
            succeeded=bool(self._successes),
            best=best,
            all_successes=list(self._successes),
            iterations=self._iter,
            elites=self._grid.size(),
            api_calls=self._api_calls,
            total_scored=len(self._history),
            usage=usage,
        )
