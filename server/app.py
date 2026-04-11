"""
CAST – Server entry point for OpenEnv multi-mode deployment.
This module is referenced by pyproject.toml [project.scripts]:
    server = "server.app:main"
"""

import uvicorn
import random
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
    """OpenEnv required: Reset the environment to a random initial state."""
    state = {
        "gesture": random.choice(GESTURES),
        "noise":   random.choice(NOISES),
        "context": random.choice(CONTEXTS),
    }
    return JSONResponse(content=state)

@app.post("/step")
async def step(request: Request):
    """OpenEnv required: Given a state dict, the agent takes a step."""
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
# Main entry point (required by OpenEnv)
# ──────────────────────────────────────────────
def main():
    """
    Server entry point for OpenEnv multi-mode deployment.
    Referenced as: server = "server.app:main" in pyproject.toml
    """
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
