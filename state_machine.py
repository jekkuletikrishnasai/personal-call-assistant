import os
import re
import json
import logging
import httpx
from typing import Optional, Dict, Any, List

logger = logging.getLogger("state_machine")
logging.basicConfig(level=logging.INFO)

class CallStateMachine:
    def __init__(self, caller_number: str, contact_name: Optional[str] = None):
        self.caller_number = caller_number
        self.contact_name = contact_name.strip() if contact_name and contact_name.strip() else None
        self.locked_language: Optional[str] = None  # 'te', 'hi', or 'en'
        self.history: List[Dict[str, Any]] = []
        self.turn_count = 0
        self.is_completed = False

    def detect_and_lock_language(self, caller_text: str) -> str:
        """Detects language on Turn 1 and permanently locks it."""
        has_telugu_script = bool(re.search(r'[\u0C00-\u0C7F]', caller_text))
        telugu_words = ["namaskaram", "bagunnara", "cheppandi", "telugu", "enti", "kavalandi", "matladali"]
        is_telugu = has_telugu_script or any(w in caller_text.lower() for w in telugu_words)

        has_hindi_script = bool(re.search(r'[\u0900-\u097F]', caller_text))
        hindi_words = ["namaste", "kaise ho", "bataye", "hindi", "kya", "bhai", "haan", "baat"]
        is_hindi = has_hindi_script or any(w in caller_text.lower() for w in hindi_words)

        if is_telugu:
            self.locked_language = "te"
        elif is_hindi:
            self.locked_language = "hi"
        else:
            self.locked_language = "en"

        return self.locked_language

    def _build_instructions(self) -> str:
        lang_names = {
            "te": "Telugu (తెలుగు)",
            "hi": "Hindi (हिंदी)",
            "en": "English"
        }
        current_lang = lang_names.get(self.locked_language, "English")

        name_instruction = (
            f"The caller is '{self.contact_name}'. Greet them politely by name. DO NOT ask who they are."
            if self.contact_name
            else "The caller's name is UNKNOWN. Ask for their full name politely."
        )

        return (
            f"Role: Automated personal call assistant for the user.\n"
            f"Rule 1: Speak 100% in {current_lang}. Never switch languages.\n"
            f"Rule 2: {name_instruction}\n"
            f"Rule 3: Keep responses under 2 short sentences (strictly <= 25 words).\n"
            f"Rule 4: Inquire about the purpose of the call and degree of urgency.\n"
            f"Rule 5: If asked to connect directly to the user, say the user is currently unavailable and offer to take a message.\n"
            f"Rule 6: Summarize details before ending, say thanks and goodbye.\n"
            f"Rule 7: Output natural conversational text only. No markdown, asterisks, or bullet points."
        )

    async def process_turn(self, caller_text: str) -> str:
        self.turn_count += 1

        if not self.locked_language:
            self.detect_and_lock_language(caller_text)

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.error("GEMINI_API_KEY is not set in Render environment variables.")
            return self._fallback_reply()

        # Build clean conversation history
        self.history.append({"role": "user", "parts": [{"text": caller_text}]})

        instructions = self._build_instructions()
        
        # Structure payload to be universally accepted by all Gemini versions
        formatted_contents = []
        
        # Inject system instructions as the preamble in turn 1
        formatted_contents.append({
            "role": "user",
            "parts": [{"text": f"[System Context & Instructions]\n{instructions}"}]
        })
        formatted_contents.append({
            "role": "model",
            "parts": [{"text": "Understood. I will follow all instructions strictly."}]
        })
        
        # Append active call conversation
        for msg in self.history:
            formatted_contents.append({
                "role": msg["role"],
                "parts": msg["parts"]
            })

        payload = {
            "contents": formatted_contents,
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 80
            }
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=payload, timeout=12.0)
                
                if res.status_code == 200:
                    data = res.json()
                    reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    clean_reply = re.sub(r'[*_#`]', '', reply)
                    self.history.append({"role": "model", "parts": [{"text": clean_reply}]})

                    if any(w in clean_reply.lower() for w in ["goodbye", "ధన్యవాదాలు", "धन्यवाद", "take care"]):
                        self.is_completed = True

                    return clean_reply
                else:
                    logger.error(f"Gemini API returned status {res.status_code}: {res.text}")
                    return self._fallback_reply()
        except Exception as e:
            logger.error(f"HTTP connection failed: {e}")
            return self._fallback_reply()

    def _fallback_reply(self) -> str:
        if self.locked_language == "te":
            return "క్షమించండి, మీరు చెప్పింది సరిగ్గా వినిపించలేదు. దయచేసి మళ్లీ చెప్పండి."
        elif self.locked_language == "hi":
            return "माफ़ कीजिए, मैं सुन नहीं सका। क्या आप फिर से कह सकते हैं?"
        return "I apologize, could you please repeat that?"

    async def generate_bilingual_summary(self) -> Dict[str, str]:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return {"english": "Summary unavailable", "telugu": "సారాంశం అందుబాటులో లేదు"}

        transcript_text = "\n".join([
            f"{'Caller' if m['role'] == 'user' else 'AI'}: {m['parts'][0]['text']}"
            for m in self.history
        ])

        summary_prompt = f"""
Summarize this call transcript in two sections (English and Telugu):
Caller Contact Name: {self.contact_name or 'Unknown'}
Caller Phone: {self.caller_number}

Transcript:
{transcript_text}

Format exactly:
---ENGLISH---
Caller: <name>
Reason: <brief purpose>
Urgency: <Urgent/Standard>
Message: <details>

---TELUGU---
కాలర్: <name in Telugu>
కారణం: <purpose in Telugu>
ముఖ్యత: <Urgent/Standard in Telugu>
సందేశం: <details in Telugu>
"""
        payload = {
            "contents": [{"role": "user", "parts": [{"text": summary_prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 300}
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload, timeout=12.0)
            if res.status_code == 200:
                raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                parts = raw_text.split("---TELUGU---")
                english_part = parts[0].replace("---ENGLISH---", "").strip()
                telugu_part = parts[1].strip() if len(parts) > 1 else "వివరాలు నమోదు చేయబడ్డాయి."
                return {"english": english_part, "telugu": telugu_part}
            return {"english": transcript_text, "telugu": transcript_text}