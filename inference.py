"""
CAST - Communication Accessibility & Sync Tool
Inference Module: RL agent for sign language interpretation with LLM explanation.
"""

import math
import os
import sys
import time


def normalize_score(reward: float) -> float:
    """
    Map any reward value to strictly (0, 1) using sigmoid.
    sigmoid(x) = 1 / (1 + e^-x)  — never exactly 0 or 1 for finite x.
    Rounded to 4 decimal places and clamped away from boundaries.
    """
    raw = 1.0 / (1.0 + math.exp(-float(reward)))
    return round(min(max(raw, 0.0001), 0.9999), 4)

# ──────────────────────────────────────────────
# LLM Client (hackathon-injected env vars)
# ──────────────────────────────────────────────

def get_llm_client():
    """
    Returns an OpenAI-compatible client using hackathon-provided env vars:
        API_BASE_URL  — LiteLLM proxy base URL
        API_KEY       — hackathon-issued API key
    Falls back gracefully if env vars are not set (local dev).
    """
    try:
        from openai import OpenAI
        api_base = os.environ.get("API_BASE_URL", "").strip()
        api_key  = os.environ.get("API_KEY", "no-key").strip()
        if api_base:
            return OpenAI(base_url=api_base, api_key=api_key)
    except Exception:
        pass
    return None


def llm_explain(gesture: str, action: str, context: str, reward: float) -> str:
    """
    Calls the LiteLLM proxy to generate a natural-language explanation
    of the agent's decision. Falls back to a templated string if LLM
    is unavailable (e.g., local dev without API_BASE_URL set).
    """
    client = get_llm_client()
    if client is None:
        return (
            f"[LLM-FALLBACK] Agent chose '{action}' for gesture '{gesture}' "
            f"in '{context}' context (reward={reward})."
        )

    prompt = (
        f"You are an AI assistant for a sign language accessibility system.\n"
        f"The agent observed gesture='{gesture}' in context='{context}' "
        f"and chose action='{action}' earning reward={reward}.\n"
        f"In one short sentence, explain why this was the correct decision."
    )
    try:
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a concise AI explainability assistant."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=80,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"[LLM-ERROR: {str(e)[:60]}] Agent chose '{action}' for "
            f"gesture '{gesture}' in '{context}' (reward={reward})."
        )


# ──────────────────────────────────────────────
# Environment Constants
# ──────────────────────────────────────────────

GESTURES   = ["hello", "stop", "help", "danger"]
NOISES     = ["low", "medium", "high"]
CONTEXTS   = ["classroom", "road", "home"]
ACTIONS    = ["show_text", "ask_repeat", "trigger_alert"]
EMERGENCY  = {"help", "danger"}

TASK_NAME  = "sign_language_interpretation"

# ──────────────────────────────────────────────
# Structured stdout logger  (Phase 2 required)
# ──────────────────────────────────────────────

def log_start(task: str = TASK_NAME):
    print(f"[START] task={task}", flush=True)

def log_step(step: int, reward: float = 0.0, **kwargs):
    extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
    line = f"[STEP] step={step} reward={reward}"
    if extras:
        line += f" {extras}"
    print(line, flush=True)

def log_end(task: str = TASK_NAME, score: float = 0.0, steps: int = 5):
    print(f"[END] task={task} score={score} steps={steps}", flush=True)


# ──────────────────────────────────────────────
# Reward Calculator
# ──────────────────────────────────────────────

def compute_reward(gesture: str, action: str, response_time: float) -> dict:
    """
    Reward logic:
      Correct interpretation  -> +1
      Wrong                   -> -1
      Emergency handled right -> +2 bonus
      Fast response (<0.5s)   -> +0.5
    """
    reward = 0.0
    breakdown = []

    if gesture in EMERGENCY and action == "trigger_alert":
        reward += 1
        reward += 2
        breakdown.append("correct_emergency (+3)")
    elif gesture not in EMERGENCY and gesture == "stop" and action == "show_text":
        reward += 1
        breakdown.append("correct_interpretation (+1)")
    elif gesture == "hello" and action == "show_text":
        reward += 1
        breakdown.append("correct_interpretation (+1)")
    else:
        reward -= 1
        breakdown.append("wrong_interpretation (-1)")

    if response_time < 0.5:
        reward += 0.5
        breakdown.append("fast_response (+0.5)")

    return {"total": round(reward, 2), "breakdown": breakdown}


# ──────────────────────────────────────────────
# Core Inference Function
# ──────────────────────────────────────────────

def predict(input: dict) -> dict:
    """
    Accepts:
        input = {"gesture": str, "noise": str, "context": str}

    Returns:
        {"action": str, "reward": dict, "message": str, "state": dict}

    Stdout (Phase 2 format, flush=True):
        [START] task=sign_language_interpretation
        [STEP] step=1 reward=0.0 action=validate_input
        [STEP] step=2 reward=0.0 action=select_action
        [STEP] step=3 reward=<X>  action=compute_reward
        [STEP] step=4 reward=<X>  action=llm_explain
        [STEP] step=5 reward=<X>  action=build_result
        [END] task=sign_language_interpretation score=<X> steps=5
    """

    # ── [START] ───────────────────────────────
    log_start()

    # ── Step 1: Validate input ────────────────
    gesture = str(input.get("gesture", "")).lower().strip()
    noise   = str(input.get("noise",   "")).lower().strip()
    context = str(input.get("context", "")).lower().strip()

    if gesture not in GESTURES:
        gesture = "hello"
    if noise not in NOISES:
        noise = "low"
    if context not in CONTEXTS:
        context = "home"

    state = {"gesture": gesture, "noise": noise, "context": context}
    log_step(step=1, reward=0.0, action="validate_input")

    # ── Step 2: Select action ─────────────────
    start_time = time.perf_counter()

    if gesture in EMERGENCY:
        action = "trigger_alert"
    elif noise == "high":
        action = "ask_repeat"
    else:
        action = "show_text"

    response_time = time.perf_counter() - start_time
    log_step(step=2, reward=0.0, action="select_action")

    # ── Step 3: Compute reward ────────────────
    reward = compute_reward(gesture, action, response_time)
    log_step(step=3, reward=reward["total"], action="compute_reward")

    # ── Step 4: LLM explanation (calls API_BASE_URL proxy) ───────────
    message = llm_explain(gesture, action, context, reward["total"])
    log_step(step=4, reward=reward["total"], action="llm_explain")

    # ── Step 5: Build result ──────────────────
    result = {
        "action":          action,
        "reward":          reward,
        "message":         message,
        "state":           state,
        "response_time_s": round(response_time, 6),
    }
    log_step(step=5, reward=reward["total"], action="build_result")

    # ── [END] ─────────────────────────────────
    # Score MUST be strictly (0, 1) — normalize raw reward via sigmoid
    normalized = normalize_score(reward["total"])
    log_end(score=normalized, steps=5)

    return result


# ──────────────────────────────────────────────
# Quick self-test (run directly)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {"gesture": "hello",  "noise": "low",    "context": "classroom"},
        {"gesture": "danger", "noise": "medium", "context": "road"},
        {"gesture": "stop",   "noise": "high",   "context": "home"},
        {"gesture": "help",   "noise": "low",    "context": "road"},
    ]

    for tc in test_cases:
        print("\n" + "=" * 50, flush=True)
        output = predict(tc)
        print(f"Action : {output['action']}", flush=True)
        print(f"Reward : {output['reward']}", flush=True)
        print(f"Message: {output['message']}", flush=True)
