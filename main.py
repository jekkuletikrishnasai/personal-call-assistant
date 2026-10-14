import os
import logging
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from state_machine import CallStateMachine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("call-assistant-server")

app = FastAPI(title="Multilingual Personal Call Assistant")

# In-memory session store for active calls
active_sessions: dict[str, CallStateMachine] = {}

class TestCallTurnRequest(BaseModel):
    call_id: str
    caller_text: str
    caller_phone: str = "+91-XXXXXXXXXX"
    contact_name: Optional[str] = None

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
    """
    Test endpoint allowing you to test conversational turns, language locking,
    and bilingual summaries directly via HTTP requests.
    """
    # Create session if not present
    if request.call_id not in active_sessions:
        active_sessions[request.call_id] = CallStateMachine(
            caller_number=request.caller_phone,
            contact_name=request.contact_name
        )

    session = active_sessions[request.call_id]
    
    # Process caller input
    ai_reply = await session.process_turn(request.caller_text)
    
    response_payload = {
        "call_id": request.call_id,
        "turn": session.turn_count,
        "locked_language": session.locked_language,
        "ai_reply": ai_reply,
        "is_completed": session.is_completed
    }

    # If call ended, generate the dual-language summary
    if session.is_completed:
        summary = await session.generate_bilingual_summary()
        response_payload["final_summary"] = summary
        # Clean up session
        del active_sessions[request.call_id]

    return response_payload

@app.websocket("/voice-stream/{call_id}")
async def voice_stream_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    logger.info(f"Incoming call stream connected. Call ID: {call_id}")
    try:
        while True:
            audio_chunk = await websocket.receive_bytes()
    except WebSocketDisconnect:
        logger.info(f"Call {call_id} disconnected.")
    except Exception as e:
        logger.error(f"Error: {e}")
        await websocket.close()