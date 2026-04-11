---
title: CAST - Sign Language RL Environment
emoji: 🤟
colorFrom: purple
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🤟 CAST — Communication Accessibility & Sync Tool

> **A Reinforcement Learning environment for real-world sign language interpretation.**
> Built for the OpenEnv Hackathon · Offline-first · No external APIs

---

## 📌 Project Overview

CAST is an **OpenEnv-compatible Reinforcement Learning environment** that simulates the challenge of interpreting sign language gestures in uncertain real-world conditions.

An AI agent observes a **state** made up of:
- The **gesture** being performed (e.g., `hello`, `danger`)
- The **noise level** of the environment (e.g., `high`)
- The **context** where it is happening (e.g., `road`, `classroom`)

…and must choose the best **action** to respond appropriately.

---

## 🧠 Reinforcement Learning Explained (Simply)

| Concept | In CAST |
|---------|---------|
| **Environment** | The real-world scenario with gesture + noise + context |
| **Agent** | The AI making decisions |
| **State** | What the agent observes at each step |
| **Action** | What the agent does (`show_text`, `ask_repeat`, `trigger_alert`) |
| **Reward** | Score given for how good the decision was |

The agent learns over time to **maximize total reward** by associating states with the correct actions.

---

## 🗂️ Project Structure

```
cast-openenv/
├── inference.py     # Core RL agent logic + predict() function
├── openenv.yaml     # Environment specification (state, actions, rewards)
├── app.py           # FastAPI + Gradio app (REST API + UI)
├── Dockerfile       # Docker build for HF Space
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## ⚙️ State & Action Space

### State Space

| Field | Values |
|-------|--------|
| `gesture` | `hello`, `stop`, `help`, `danger` |
| `noise` | `low`, `medium`, `high` |
| `context` | `classroom`, `road`, `home` |

### Action Space

| Action | Description |
|--------|-------------|
| `show_text` | Display the gesture translation as text |
| `ask_repeat` | Ask the user to repeat (high noise) |
| `trigger_alert` | Fire an emergency alert for critical gestures |

### Reward Logic

| Event | Reward |
|-------|--------|
| Correct interpretation | **+1** |
| Wrong interpretation | **−1** |
| Emergency handled correctly | **+2 bonus** |
| Fast response (< 0.5 s) | **+0.5** |

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run inference directly

```bash
python inference.py
```

Expected output format:
```
START
STEP: Validating input state
STEP: Selecting action via rule-based policy
STEP: Computing reward signal
STEP: Building result payload
END
```

### 3. Launch the Gradio demo

```bash
python app.py
```

Then open your browser at `http://localhost:7860`

---

## 🎮 Demo Explanation

The Gradio interface lets you:
1. **Pick a gesture** (e.g., `danger`)
2. **Set noise level** (e.g., `low`)
3. **Choose a context** (e.g., `road`)
4. Click **Run Agent** to see the agent's chosen action, reward, and message in real time.

---

## 📋 Agent Decision Rules

```
if gesture in ["help", "danger"]  →  trigger_alert
elif noise == "high"              →  ask_repeat
else                              →  show_text
```

---

## 🔒 Constraints

- ✅ No external API calls
- ✅ No hardcoded API keys
- ✅ Fully offline capable
- ✅ Python 3.9+ compatible

---

## 👤 Author

**Ankit Pandit** — OpenEnv Hackathon Submission · 2026
