# Benchmarking strategy: open-source vs Opus

## What shadowing can and cannot tell you

Shadowing (send each VS Code turn to the OSS model *and* to Opus, show the user
only the OSS answer) is cheap to run and needs nothing from participants. It has
one structural limit to keep in mind when reading the numbers:

**Only the first turn of a conversation is a clean comparison.** From turn 2 on,
the context Opus sees was produced by the OSS model: its earlier answers, its
tool calls, and the tool results those calls produced. Opus is answering "what
would you do next, given what this other model already did?" In agentic modes
(Continue agent, Cline, Copilot agent) the Opus tool calls are never executed, so
you never see where Opus's own trajectory would have gone. Per-turn shadow
scores measure **next-step quality**, not **task success**.

So shadowing is one signal among three. The plan below combines them.

## Recommended setup: three complementary signals

| # | Signal | What it measures | Cost | Built here |
|---|---|---|---|---|
| 1 | **Shadow + pairwise judge** on live traffic | Per-turn quality on real hackathon prompts, sliced by task type and complexity | Opus tokens for shadow + judge | yes: `recorder`, `bench/judge.py`, `bench/report.py` |
| 2 | **Task-based eval with hidden tests** | End-to-end task success (pass@1), cost and time per solved task | Compute + Opus tokens for the Opus arm | partly: eval listener `:8081` exposes all 3 models behind one OpenAI API |
| 3 | **Human blind preference** | What developers actually prefer | Participants' time | yes: `/arena` page, votes in `report.py` |

### 1. Shadow + judge (built)

* Every chat turn is mirrored by Envoy to the `recorder`, which tags it with the
  NVIDIA prompt classifier (task type, complexity) and sends it to Opus.
* `bench/judge.py` runs a **blind, position-swapped pairwise** judge. A verdict
  counts only if both orders agree. Otherwise it is recorded as a tie, and
  `report.py` shows the inconsistency rate (a high rate means the pair is too
  close to call or the rubric is too vague).
* **Judge bias.** Opus judging Opus is subject to self-preference. Run a second
  judge from another family (`--judge-openai-url http://localhost:8081/v1
  --judge-model <oss>`) and report where the two judges agree. Hand-label about
  50 turns yourself to calibrate both judges.
* Headline metric: the **OSS win-or-tie rate** with a 95% Wilson interval, by
  model, task type, complexity bucket, and chat vs agentic. It answers the usual
  question ("is the open model good enough for this kind of work?"), and the
  complexity slice tells you where the router threshold should sit.
* Restrict the headline number to **first turns** (`n_messages` ≤ 2 after the
  system prompt) and report later turns separately, for the reason above.

### 2. Task-based eval (recommended main result)

Prepare 15–30 small, self-contained coding tasks that fit the hackathon theme.
Each task has a repo snapshot, an instruction, and **hidden tests**. Then:

* **Offline arm (organizers):** run the same agent harness (for example
  mini-SWE-agent, OpenHands, or aider's benchmark mode) against
  `http://localhost:8081/v1` with model = `qwen3-coder`, `gpt-oss`, and
  `claude-opus-5-5`. Same harness, same prompts, same tool set: only the model
  changes. Run each task 3–5 times per model; report pass@1, the variance
  between runs, tokens per solved task, and wall-clock time.
* **Live arm (teams):** give teams some of those tasks as warm-up challenges.
  Their hidden-test pass rate, time to green, and number of turns are a
  real-world complement to the offline numbers.

This is the comparison that holds up when someone challenges it, because the
grader is a test suite, not a model.

### 3. Human blind preference (built)

`http://<gateway>:8080/arena` shows a random first-turn pair. The two answers
appear left and right in random order, with no model names, and people vote with
the keys 1 (left better), 2 (right better), 3 (about the same), 4 (both wrong)
or S (skip).

* **Blinding:** which side holds which model is stored on the server, per
  ballot; the page never learns it. Pairs where an answer names its maker ("I'm
  Claude", "as Qwen…") are never shown, and `<think>` blocks are stripped.
* **Who reviews what:** with `ARENA_SCOPE=team` (default), people only see
  prompts from their own team: they know the context best, and no team's code
  leaks to another. `ARENA_SCOPE=all` mixes every team's prompts.
* **One vote per person per pair:** the team key comes through Envoy, and each
  browser gets its own id, so several people on a team vote separately.
  Pairs with the fewest votes are shown first.
* **Results:** `report.py` adds a "Human blind votes" table with the OSS
  better-or-same rate and its 95% CI. "Both wrong" votes are counted but left
  out of that rate.
* **Known limit:** style can still give a model away (Opus's formatting habits,
  for example). Compare the human results with the judge results per slice; a
  big gap means one of them is picking up style, not substance.

Ask participants to vote on 10 pairs during breaks.

## Experimental hygiene for the hackathon

* **Don't let teams choose the model freely.** Self-selection confounds the
  model with team skill and task type. Assign teams to model A or B at random,
  and swap them halfway through (a crossover design). Then each team's data
  covers both models and team skill cancels out.
* **Tell participants their prompts go to Anthropic.** Shadowing sends their
  code to a third-party API. Put it in the rules and offer an opt-out key
  (`SHADOW_RATE=0`, or a team list the recorder skips).
* **Budget.** Agentic sessions resend the whole context on every turn. Use
  `SHADOW_RATE` (for example 0.3) on the agentic modes if cost matters, and run
  the judge after the event with `--limit`. The Message Batches API is a 50%
  cheaper option for that offline judging pass.
* **Same token budget and settings.** Record `SHADOW_EFFORT` and each vLLM
  model's settings next to the results. OSS reasoning models (gpt-oss) also have
  a reasoning-effort knob, so compare at stated settings.
* **Refusal fallbacks.** The shadow uses `fallbacks: "default"`. A refused turn
  is answered by another Claude model; those turns are recorded with
  `shadow_served_model` and excluded by `judge.py`.
