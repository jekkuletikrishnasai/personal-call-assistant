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

    def _build_system_prompt(self) -> str:
        lang_names = {
            "te": "Telugu (తెలుగు)",
            "hi": "Hindi (हिंदी)",
            "en": "English"
        }
        current_lang = lang_names.get(self.locked_language, "English")

        name_instruction = (
            f"The caller's identity is already known: '{self.contact_name}'. Greet them politely by name and do NOT ask for their name."
            if self.contact_name
            else "The contact name is UNKNOWN. You MUST politely ask for their name during the conversation."
        )

        return f"""
You are an automated personal phone call assistant that professionally answers incoming calls on behalf of the user.

BEHAVIORS AND RULES:
1. LANGUAGE LOCK: Speak 100% in {current_lang}. Maintain all subsequent conversation strictly in this language.
2. CALLER IDENTITY: {name_instruction}
3. CONVERSATIONAL CADENCE: Keep conversational turns concise, clear, and direct to emulate a natural voice interaction. Strictly under 2 short sentences (maximum 25 words).
4. HANDLING INQUIRIES & DIRECT CONNECTION:
   - Politely inquire about the purpose of the call.
   - Gather essential details: caller's name (if not known), contact info, reason for calling, and degree of urgency (Urgent vs Standard).
   - If asked to connect directly to the user or check availability, say:
     * Telugu: "వారు ప్రస్తుతం అందుబాటులో లేరు. నేను మీ ముఖ్యమైన సమాచారాన్ని నమోదు చేసి వెంటనే వారికి తెలియజేస్తాను."
     * Hindi: "वे अभी उपलब्ध नहीं हैं। मैं आपका संदेश नोट कर लेता हूँ और उन्हें तुरंत सूचित कर दूँगा।"
     * English: "The user is currently unavailable. I would be glad to take a detailed message and notify them immediately."
5. MESSAGE RECORDING & WRAP-UP:
   - Summarize the key points back to the caller in {current_lang} to confirm accuracy.
   - Thank the caller in {current_lang} and conclude the call in a courteous manner (e.g., goodbye, ధన్యవాదాలు, धन्यवाद).
6. NO MARKDOWN: Output conversational spoken words only. Never use asterisks, lists, bullet points, or emojis.
"""

    async def process_turn(self, caller_text: str) -> str:
        self.turn_count += 1

        if not self.locked_language:
            self.detect_and_lock_language(caller_text)

        self.history.append({"role": "user", "parts": [{"text": caller_text}]})

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.error("GEMINI_API_KEY is not set in environment variables.")
            return self._fallback_reply()

        prompt = self._build_system_prompt()
        payload = {
            "system_instruction": {"parts": [{"text": prompt}]},
            "contents": self.history,
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 80
            }
        }

        # Updated to gemini-1.5-flash to eliminate 404 errors
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=payload, timeout=10.0)
                
                if res.status_code == 200:
                    data = res.json()
                    reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    clean_reply = re.sub(r'[*_#`]', '', reply)
                    self.history.append({"role": "model", "parts": [{"text": clean_reply}]})

                    if any(w in clean_reply.lower() for w in ["goodbye", "ధన్యవాదాలు", "धन्यवाद", "take care"]):
                        self.is_completed = True

                    return clean_reply
                else:
                    logger.error(f"Gemini API Error [{res.status_code}]: {res.text}")
                    return self._fallback_reply()
        except Exception as e:
            logger.error(f"HTTP Request Exception: {e}")
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
Analyze this call transcript and produce a structured summary in English and Telugu:
Caller Contact Name: {self.contact_name or 'Unknown'}
Caller Phone: {self.caller_number}

Transcript:
{transcript_text}

Format your output exactly as:
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
            res = await client.post(url, json=payload, timeout=10.0)
            if res.status_code == 200:
                raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                parts = raw_text.split("---TELUGU---")
                english_part = parts[0].replace("---ENGLISH---", "").strip()
                telugu_part = parts[1].strip() if len(parts) > 1 else "వివరాలు నమోదు చేయబడ్డాయి."
                return {"english": english_part, "telugu": telugu_part}
            return {"english": transcript_text, "telugu": transcript_text}