import os
import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from state_machine import CallStateMachine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("call-assistant-server")

# In-memory session store for active calls
active_sessions: dict[str, CallStateMachine] = {}

class TestCallTurnRequest(BaseModel):
    call_id: str
    caller_text: str
    caller_phone: str = "+91-XXXXXXXXXX"
    contact_name: Optional[str] = None

# Background worker that pings this service every 40 seconds to prevent Render spin-down
async def keep_alive_task():
    await asyncio.sleep(15)  # Let server complete startup
    app_url = "https://personal-call-assistant.onrender.com/health"
    async with httpx.AsyncClient() as client:
        while True:
            try:
                res = await client.get(app_url, timeout=5.0)
                logger.info(f"Keep-alive self-ping status: {res.status_code}")
            except Exception as e:
                logger.warning(f"Keep-alive ping failed: {e}")
            await asyncio.sleep(40)  # Ping safely inside Render's idle threshold

@asynccontextmanager
async def lifespan(app: FastAPI):
    pinger_task = asyncio.create_task(keep_alive_task())
    yield
    pinger_task.cancel()

app = FastAPI(title="Multilingual Personal Call Assistant", lifespan=lifespan)

@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={"status": "online", "message": "Voice assistant server is active"}
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/test-turn")
async def test_turn(request: TestCallTurnRequest):
    if request.call_id not in active_sessions:
        active_sessions[request.call_id] = CallStateMachine(
            caller_number=request.caller_phone,
            contact_name=request.contact_name
        )

    session = active_sessions[request.call_id]
    ai_reply = await session.process_turn(request.caller_text)

    response_payload = {
        "call_id": request.call_id,
        "turn": session.turn_count,
        "locked_language": session.locked_language,
        "ai_reply": ai_reply,
        "is_completed": session.is_completed
    }

    if session.is_completed:
        summary = await session.generate_bilingual_summary()
        response_payload["final_summary"] = summary
        del active_sessions[request.call_id]

    return response_payload

@app.websocket("/voice-stream/{call_id}")
async def voice_stream_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    logger.info(f"Call stream connected: {call_id}")
    try:
        while True:
            await websocket.receive_bytes()
    except WebSocketDisconnect:
        logger.info(f"Call disconnected: {call_id}")
    except Exception as e:
        logger.error(f"Stream error: {e}")
        await websocket.close()