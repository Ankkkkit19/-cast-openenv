"""
CAST – FastAPI + Gradio App
Exposes OpenEnv-required REST endpoints (/reset, /step, /predict)
and a Gradio UI mounted at /ui
"""

import json
import random
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import gradio as gr
from inference import predict

# ──────────────────────────────────────────────
# Environment constants
# ──────────────────────────────────────────────
GESTURES = ["hello", "stop", "help", "danger"]
NOISES   = ["low", "medium", "high"]
CONTEXTS = ["classroom", "road", "home"]

# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="CAST OpenEnv API",
    description="Sign Language RL Environment — OpenEnv compatible REST API",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {
        "name": "CAST",
        "description": "Communication Accessibility & Sync Tool — RL Environment",
        "status": "running",
        "endpoints": ["/reset", "/step", "/predict"],
    }

@app.post("/reset")
async def reset():
    """
    OpenEnv required: Reset the environment to a random initial state.
    Returns the initial observation (state).
    """
    state = {
        "gesture": random.choice(GESTURES),
        "noise":   random.choice(NOISES),
        "context": random.choice(CONTEXTS),
    }
    return JSONResponse(content=state)

@app.post("/step")
async def step(request: Request):
    """
    OpenEnv required: Given a state dict, the agent takes a step.
    Returns action, reward, next_state, done.
    """
    data   = await request.json()
    result = predict(data)
    return JSONResponse(content={
        "observation": result["state"],
        "action":      result["action"],
        "reward":      result["reward"]["total"],
        "reward_info": result["reward"],
        "message":     result["message"],
        "done":        False,
    })

@app.post("/predict")
async def predict_endpoint(request: Request):
    """Direct predict — returns full inference output."""
    data   = await request.json()
    result = predict(data)
    return JSONResponse(content=result)

# ──────────────────────────────────────────────
# Gradio UI (mounted at /ui)
# ──────────────────────────────────────────────
def run_cast(gesture: str, noise: str, context: str) -> str:
    result = predict({"gesture": gesture, "noise": noise, "context": context})
    return json.dumps(result, indent=2)

with gr.Blocks(
    title="CAST – Sign Language RL Environment",
    theme=gr.themes.Soft(primary_hue="violet"),
) as demo:
    gr.Markdown(
        """
        # 🤟 CAST – Communication Accessibility & Sync Tool
        **A Reinforcement Learning environment for real-world sign language interpretation.**

        Select the current environment state and click **Run Agent** to see the
        agent's action, reward, and decision message.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            gesture_dd = gr.Dropdown(
                choices=GESTURES, value="hello",
                label="🖐️ Gesture",
                info="The sign language gesture being performed.",
            )
            noise_dd = gr.Dropdown(
                choices=NOISES, value="low",
                label="🔊 Noise Level",
                info="Environmental noise that may affect gesture clarity.",
            )
            context_dd = gr.Dropdown(
                choices=CONTEXTS, value="classroom",
                label="🏫 Context",
                info="Where the interaction is taking place.",
            )
            run_btn = gr.Button("▶  Run Agent", variant="primary")
        with gr.Column(scale=2):
            output_box = gr.Code(
                label="📦 Agent Output (JSON)",
                language="json",
                lines=18,
            )

    gr.Markdown(
        """
        ---
        ### 📖 How the Agent Decides
        | Condition | Action |
        |-----------|--------|
        | Gesture = `help` or `danger` | `trigger_alert` |
        | Noise = `high` | `ask_repeat` |
        | Everything else | `show_text` |

        ### 🏆 Reward Table
        | Event | Reward |
        |-------|--------|
        | Correct interpretation | +1 |
        | Wrong interpretation | –1 |
        | Emergency handled correctly | +2 bonus |
        | Fast response (< 0.5 s) | +0.5 |

        ---
        **REST API Endpoints:** `POST /reset` · `POST /step` · `POST /predict`
        """
    )

    run_btn.click(fn=run_cast, inputs=[gesture_dd, noise_dd, context_dd], outputs=output_box)
    demo.load(fn=run_cast, inputs=[gesture_dd, noise_dd, context_dd], outputs=output_box)

# Mount Gradio at /ui — FastAPI handles /reset /step /predict directly
app = gr.mount_gradio_app(app, demo, path="/")


# ──────────────────────────────────────────────
# Server entry point (required for OpenEnv multi-mode deployment)
# ──────────────────────────────────────────────
def serve():
    """Entry point for [project.scripts] server = 'app:serve'"""
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
