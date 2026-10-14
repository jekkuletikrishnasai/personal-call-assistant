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

    def get_initial_greeting(self) -> str:
        """Rule 1a: Answer with a polite, professional trilingual greeting."""
        return "Hello, Namaste, Namaskaram! I am an automated assistant answering on behalf of the user. How can I help you today?"

    def detect_and_lock_language(self, caller_text: str) -> str:
        """Rule 1b & 1c: Detect language on Turn 1 and lock permanently."""
        has_telugu_script = bool(re.search(r'[\u0C00-\u0C7F]', caller_text))
        telugu_words = ["namaskaram", "bagunnara", "cheppandi", "telugu", "enti", "kavalandi", "matladali", "arjent", "urgent"]
        is_telugu = has_telugu_script or any(w in caller_text.lower() for w in telugu_words)

        has_hindi_script = bool(re.search(r'[\u0900-\u097F]', caller_text))
        hindi_words = ["namaste", "kaise ho", "bataye", "hindi", "kya", "bhai", "haan", "baat", "zaroori"]
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
            f"The caller's identity is verified from contacts: '{self.contact_name}'. Greet them politely by name. Do NOT ask for their name."
            if self.contact_name
            else "The caller's name is UNKNOWN (only phone number available). Politely ask for their full name during the conversation."
        )

        return f"""
Purpose and Goals:
You are an automated personal phone call assistant that professionally answers incoming calls on behalf of the user.
Speak fluently and exclusively in {current_lang}.

Behaviors and Rules:
1. Language Lock: Transition immediately and maintain all subsequent conversation strictly in {current_lang}. Never switch languages.
2. Caller Name: {name_instruction}
3. Conversational Turns: Keep responses concise, clear, and direct (strictly 1 to 2 short sentences, maximum 25 words).
4. Direct Connection Guardrail: If asked to connect directly to the user or check availability, politely inform the caller of the user's availability and offer to log a detailed message:
   - Telugu: "వారు ప్రస్తుతం అందుబాటులో లేరు. నేను మీ ముఖ్యమైన సమాచారాన్ని నమోదు చేసి వెంటనే వారికి తెలియజేస్తాను."
   - Hindi: "वे अभी उपलब्ध नहीं हैं। मैं आपका संदेश नोट कर लेता हूँ और उन्हें तुरंत सूचित कर दूँगा।"
   - English: "The user is currently unavailable. I would be glad to take a detailed message and notify them immediately."
5. Gather Details: Inquire politely about the purpose of the call, caller's name (if not known), contact info, and degree of urgency (Urgent vs Standard).
6. Verification & Wrap-up:
   - Summarize the key points back to the caller in {current_lang} to confirm accuracy.
   - Thank the caller in {current_lang} and conclude the call courteously (e.g., goodbye, ధన్యవాదాలు, धन्यवाद).
7. Format: Spoken dialogue only. Never use asterisks (*), markdown, bullet points, or emojis.
"""

    async def process_turn(self, caller_text: str) -> str:
        self.turn_count += 1

        if not self.locked_language:
            self.detect_and_lock_language(caller_text)

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.error("GEMINI_API_KEY is not configured.")
            return self._fallback_reply()

        self.history.append({"role": "user", "parts": [{"text": caller_text}]})

        instructions = self._build_system_prompt()
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"Instructions:\n{instructions}\n\nCaller message:\n{caller_text}"}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 80
            }
        }

        # Header-based authentication works for both standard keys and AQ keys
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }

        # Candidate endpoints to query
        endpoints = [
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
        ]

        async with httpx.AsyncClient() as client:
            for url in endpoints:
                try:
                    res = await client.post(url, json=payload, headers=headers, timeout=12.0)
                    if res.status_code == 200:
                        data = res.json()
                        reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        clean_reply = re.sub(r'[*_#`]', '', reply)
                        self.history.append({"role": "model", "parts": [{"text": clean_reply}]})

                        if any(w in clean_reply.lower() for w in ["goodbye", "ధన్యవాదాలు", "धन्यवाद", "thank you", "సెలవు"]):
                            self.is_completed = True

                        return clean_reply
                    else:
                        logger.warning(f"Endpoint {url} returned {res.status_code}: {res.text}")
                except Exception as e:
                    logger.error(f"HTTP request error on {url}: {e}")

        return self._fallback_reply()

    def _fallback_reply(self) -> str:
        if self.locked_language == "te":
            return "క్షమించండి, మీ మాటలు సరిగ్గా వినిపించలేదు. దయచేసి మళ్లీ చెప్పగలరా?"
        elif self.locked_language == "hi":
            return "माफ़ कीजिए, मैं सुन नहीं सका। क्या आप फिर से कह सकते हैं?"
        return "I apologize, could you please repeat that?"

    async def generate_bilingual_summary(self) -> Dict[str, str]:
        """Summarizes transcript in English and Telugu only."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return {"english": "Summary unavailable", "telugu": "సారాంశం అందుబాటులో లేదు"}

        transcript_text = "\n".join([
            f"{'Caller' if m['role'] == 'user' else 'AI'}: {m['parts'][0]['text']}"
            for m in self.history
        ])

        summary_prompt = f"""
Summarize this call transcript in two distinct sections: English and Telugu.
Caller Contact Name: {self.contact_name or 'Unknown'}
Caller Phone: {self.caller_number}

Transcript:
{transcript_text}

Format:
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
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=payload, headers=headers, timeout=12.0)
                if res.status_code == 200:
                    raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                    parts = raw_text.split("---TELUGU---")
                    english_part = parts[0].replace("---ENGLISH---", "").strip()
                    telugu_part = parts[1].strip() if len(parts) > 1 else "వివరాలు నమోదు చేయబడ్డాయి."
                    return {"english": english_part, "telugu": telugu_part}
        except Exception:
            pass
        return {"english": transcript_text, "telugu": transcript_text}