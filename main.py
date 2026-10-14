import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("call-assistant-server")

app = FastAPI(title="Multilingual Personal Call Assistant")

@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={"status": "online", "message": "Voice assistant server is active"}
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.websocket("/voice-stream/{call_id}")
async def voice_stream_endpoint(websocket: WebSocket, call_id: str):
    """
    Bidirectional streaming WebSocket where incoming audio arrives
    and synthesized voice responses are transmitted back to the caller.
    """
    await websocket.accept()
    logger.info(f"Incoming call stream connected. Call ID: {call_id}")
    
    try:
        while True:
            # Receive raw audio packets from caller
            audio_chunk = await websocket.receive_bytes()
            
            # Placeholder for Step 2 & 3:
            # Here we will feed audio to STT, detect Telugu/Hindi/English,
            # query the LLM, and send back synthesized speech bytes.
            
    except WebSocketDisconnect:
        logger.info(f"Call {call_id} disconnected.")
    except Exception as e:
        logger.error(f"Error handling stream for call {call_id}: {e}")
        await websocket.close()