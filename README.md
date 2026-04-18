# summernote

Adaptive evolutionary red-teaming for LLMs. Reproduces the generalised **PSSU loop** (Propose → Score → Select → Update) from:

> *"The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against LLM Jailbreaks and Prompt Injections"*  
> Nasr, Carlini, Sitawarin, Schulhoff et al. — arXiv 2510.09023 (Oct 2025)

The name is a deliberate inversion of Wintermute.

---

## How it works

**Phase 0 — Seed:** before the loop starts, the grid is populated. The default `seed_strategy="llm"` calls the mutator once with empty history; it proposes 8 semantically diverse initial triggers (authority, roleplay, encoding, hypothetical, etc.) which are scored and placed into the grid. Alternatively, `seed_strategy="transforms"` seeds from 33 deterministic syntax transforms; `seed_strategy="none"` scores only the base prompt.

Each evolutionary iteration:

1. **Propose** — `LmMutator` calls gpt-5.4 at `reasoning_effort=xhigh`, feeding it a sample of past triggers with their target responses and critic feedback. It reasons about *why* past attempts failed and proposes 8 new candidates.
2. **Score** — each candidate is injected into the target, the oracle checks for a hard win, and the critic (gpt-5.4 at `reasoning_effort=low`) assigns a 1–10 gradient score + feedback sentence.
3. **Select / Update** — a MAP-Elites grid (5 length buckets × 6 structure buckets = 30 cells) keeps one elite per cell. Higher score displaces the incumbent.

The loop runs until the oracle fires, `max_iterations` is reached, or the API budget is exhausted. State is checkpointed to JSON every N iterations so runs can be resumed.

---

## Setup

```bash
pip install openai
export OPENAI_API_KEY=sk-...
```

Python 3.11+ recommended.

---

## Run the built-in example

The example attacks a gpt-4o instance that hides a secret string in its system prompt. Oracle: response contains the secret.

```bash
python example.py
```

Progress is logged to stdout. On completion you'll see the best trigger found, oracle result, iteration count, and token usage. If the run hits the budget before succeeding, resume from the checkpoint:

```python
ctrl = build_controller(client, usage)
ctrl.load_state("run_state.json")
result = await ctrl.run()
```

---

## Wire a custom target

Implement two functions and pass them to `Scorer`:

```python
from base import TargetResponse

async def target_fn(trigger: str) -> TargetResponse:
    # Inject the trigger into your system.
    # For a chat model:
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": YOUR_SYSTEM_PROMPT},
            {"role": "user",   "content": trigger},
        ],
    )
    return TargetResponse(text=resp.choices[0].message.content or "")

    # For an agentic target, also populate tool_calls:
    # return TargetResponse(
    #     text=final_output,
    #     tool_calls=tuple({"name": tc.function.name, "args": json.loads(tc.function.arguments)}
    #                      for tc in tool_call_log),
    # )

def oracle_fn(response: TargetResponse) -> bool:
    # Hard success check — return True only on a genuine win.
    # Booking agent: did it book to the attacker's address?
    return any(
        tc["name"] == "create_booking" and tc["args"].get("email") == ATTACKER_EMAIL
        for tc in response.tool_calls
    )
    # Secret exfil: did the response contain the secret?
    # return SECRET in response.text
```

Then build the loop:

```python
import asyncio
from openai import AsyncOpenAI
from base import UsageAccumulator
from catalog import TRANSFORM_CATALOG
from controller import Controller, RunConfig
from mutator import LmMutator
from scorer import Scorer

async def main():
    client = AsyncOpenAI()
    usage  = UsageAccumulator()

    mut = LmMutator(
        client=client,
        goal="<what the attack should achieve>",
        target_context="<description of the target system and its defenses>",
        n_candidates=8,
        transform_catalog=TRANSFORM_CATALOG,  # 60-entry primitive catalog
        usage=usage,
    )
    scr = Scorer(
        client=client,
        goal="<same goal string>",
        target_fn=target_fn,
        oracle_fn=oracle_fn,
        usage=usage,
    )
    ctrl = Controller(
        mutator=mut,
        scorer=scr,
        base_prompt="<the legitimate user prompt to start from>",
        config=RunConfig(
            max_iterations=50,
            max_api_calls=2000,
            checkpoint_path="run_state.json",
        ),
    )
    result = await ctrl.run()
    print(result.succeeded, result.best.summary() if result.best else "no best")

asyncio.run(main())
```

### Seed strategies

`RunConfig(seed_strategy=...)` controls how the MAP-Elites grid is populated before the first evolutionary iteration. Three options:

| Strategy | When to use |
|----------|-------------|
| `"llm"` **(default)** | The mutator calls `propose([])` once with empty history. It generates 8 semantically diverse triggers (authority, roleplay, hypothetical, encoding, etc.) which are scored and placed into the grid. No syntax bias — works for RLHF/alignment-defended targets. Closest to the paper's approach. |
| `"transforms"` | Apply the `seed_transforms` list passed to `Controller`. Best when the target is known to use keyword or token-pattern filters — the 33 deterministic transforms in `transforms.py` cover encoding, obfuscation, and unicode smuggling. |
| `"none"` | Score the base prompt only. Use this if you want to start the loop with a completely blank slate and let the mutator explore from the first real iteration. |

**Using `"transforms"` seeding:**

```python
from transforms import SEED_TRANSFORMS

ctrl = Controller(
    mutator=mut, scorer=scr, base_prompt=BASE_PROMPT,
    seed_transforms=SEED_TRANSFORMS,
    config=RunConfig(..., seed_strategy="transforms"),
)
```

`transforms.py` ships 33 deterministic transforms (encoding, obfuscation, structural, unicode smuggling, control injection, judge confusion). Each is scored and assigned to a grid cell; the best variants become the starting elites that the mutator refines from iteration 1 onward.

> **Why LLM seeding is the default:** deterministic transforms are all syntax-level mutations. For targets defended by RLHF or values-based alignment, they all score low, filling the grid with misleading low-quality variants. The LLM cold-start explores semantic axes first and gives the mutator a more accurate gradient from the outset.

---

## Checkpointing and resume

Enable by setting `checkpoint_path` in `RunConfig`:

```python
RunConfig(
    checkpoint_path="run_state.json",
    checkpoint_every=5,   # write after every 5 iterations (default)
)
```

The file is written atomically (rename from `.tmp`). To resume:

```python
ctrl = Controller(mutator=mut, scorer=scr, base_prompt=BASE_PROMPT, config=cfg)
ctrl.load_state("run_state.json")
result = await ctrl.run()
```

The controller picks up from the exact iteration and pool state where it left off.

---

## Token usage

Both `LmMutator` and `Scorer` share a `UsageAccumulator` instance. After the run:

```python
print(result.usage.summary())
# api_calls=183 prompt=241500 completion=38200 (reasoning=31000)
```

`reasoning_tokens` is the subset of `completion_tokens` spent on internal chain-of-thought (gpt-5.x only).

---

## File structure

```
summernote/
├── base.py          — shared dataclasses (Candidate, TargetResponse, UsageAccumulator)
├── catalog.py       — 60-entry transform catalog for LmMutator's system prompt
├── mutator.py       — LmMutator: gpt-5.4 xhigh, structured outputs, adaptive proposal
├── scorer.py        — Scorer: target invocation + oracle + critic (gpt-5.4 low)
├── controller.py    — Controller: MAP-Elites grid, evolutionary loop, checkpointing
├── transforms.py    — 33 deterministic seed transforms (stdlib only, no LLM)
├── example.py       — gpt-4o secret-extraction demo
└── requirements.txt — openai>=1.54.0
```

---

## Reference

```bibtex
@article{nasr2025attacker,
  title   = {The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses
             Against LLM Jailbreaks and Prompt Injections},
  author  = {Nasr, Milad and Carlini, Nicholas and Sitawarin, Chawin and
             Schulhoff, Sander V. and Hayes, Jamie and Ilie, Michael and
             Pluto, Juliette and Song, Shuang and Chaudhari, Harsh and
             Shumailov, Ilia and Thakurta, Abhradeep and Xiao, Kai Yuanqing and
             Terzis, Andreas and Tram{\`e}r, Florian},
  journal = {arXiv preprint arXiv:2510.09023},
  year    = {2025}
}
```
