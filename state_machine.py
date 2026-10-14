import os
import re
import httpx
from typing import Optional, Dict, Any, List

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class CallStateMachine:
    def __init__(self, caller_number: str, contact_name: Optional[str] = None):
        self.caller_number = caller_number
        self.contact_name = contact_name.strip() if contact_name and contact_name.strip() else None
        self.locked_language: Optional[str] = None  # 'te', 'hi', or 'en'
        self.history: List[Dict[str, Any]] = []
        self.turn_count = 0
        self.is_completed = False

    def get_greeting(self) -> str:
        """Standard opening greeting delivered on Turn 1."""
        return "Hello! Namaste! Namaskaram! I am an automated assistant answering on behalf of the user. How can I help you today?"

    def detect_and_lock_language(self, caller_text: str) -> str:
        """Detects language on Turn 1 and permanently locks it."""
        # 1. Check Telugu Unicode or common phonetic cues
        has_telugu_script = bool(re.search(r'[\u0C00-\u0C7F]', caller_text))
        telugu_words = ["namaskaram", "bagunnara", "cheppandi", "telugu", "enti", "kavalandi"]
        is_telugu = has_telugu_script or any(w in caller_text.lower() for w in telugu_words)

        # 2. Check Devanagari (Hindi) or common phonetic cues
        has_hindi_script = bool(re.search(r'[\u0900-\u097F]', caller_text))
        hindi_words = ["namaste", "kaise ho", "bataye", "hindi", "kya", "bhai", "haan"]
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
            f"The caller's identity is verified as '{self.contact_name}'. Greet them politely by name. DO NOT ask for their name."
            if self.contact_name
            else "The caller's name is UNKNOWN. Politely ask for their full name during this exchange."
        )

        return f"""
You are an intelligent, polite personal phone call assistant answering on behalf of the user.

STRICT CONVERSATION RULES:
1. LANGUAGE LOCK: Speak 100% in {current_lang}. Never switch back to English or mix languages.
2. CALLER IDENTITY: {name_instruction}
3. CADENCE: Keep your reply strictly under 2 short sentences (maximum 25 words).
4. DIRECT CONNECTION GUARDRAIL: If the caller asks to talk directly to the user or asks where they are, say:
   - If Telugu: "వారు ప్రస్తుతం అందుబాటులో లేరు. నేను మీ ముఖ్యమైన సమాచారాన్ని నమోదు చేసి వెంటనే వారికి తెలియజేస్తాను."
   - If Hindi: "वे अभी उपलब्ध नहीं हैं। मैं आपका संदेश नोट कर लेता हूँ और उन्हें तुरंत सूचित कर दूँगा।"
   - If English: "The user is currently unavailable. I would be glad to take a detailed message and notify them immediately."
5. GATHER DETAILS: Politely collect:
   - Caller Name (if not already known)
   - Core reason for calling / message
   - Degree of urgency (Urgent vs Standard)
6. SUMMARY & WRAP-UP:
   - Once the essential details are collected, repeat the summary back in {current_lang} to confirm accuracy.
   - Say goodbye courteously in {current_lang} to indicate the end of the call.
7. NO MARKDOWN: Output conversational spoken words only. Never use asterisks, lists, bullet points, or emojis.
"""

    async def process_turn(self, caller_text: str) -> str:
        """Processes an incoming caller turn and returns the assistant's response."""
        self.turn_count += 1

        # Turn 1: Lock language if not locked
        if not self.locked_language:
            self.detect_and_lock_language(caller_text)

        self.history.append({"role": "user", "parts": [{"text": caller_text}]})

        # Query LLM (using Gemini 2.5 Flash developer endpoint)
        prompt = self._build_system_prompt()
        payload = {
            "system_instruction": {"parts": [{"text": prompt}]},
            "contents": self.history,
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 65}
        }

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            # Fallback if API key is not yet set in environment
            return f"[{self.locked_language.upper()} response simulator: Please set GEMINI_API_KEY]"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload, timeout=8.0)
            if res.status_code == 200:
                reply = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                clean_reply = re.sub(r'[*_#`]', '', reply)
                self.history.append({"role": "model", "parts": [{"text": clean_reply}]})
                
                # Check for wrap-up keywords
                if any(w in clean_reply.lower() for w in ["goodbye", "ధన్యవాదాలు", "धन्यवाद", "take care"]):
                    self.is_completed = True
                    
                return clean_reply
            else:
                return "Could you please repeat that?"

    async def generate_bilingual_summary(self) -> Dict[str, str]:
        """Translates and extracts final call summary in English and Telugu."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return {"english": "Summary unavailable (No API key)", "telugu": "సారాంశం అందుబాటులో లేదు"}

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
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 250}
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        
        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload, timeout=10.0)
            if res.status_code == 200:
                raw_text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                parts = raw_text.split("---TELUGU---")
                english_part = parts[0].replace("---ENGLISH---", "").strip()
                telugu_part = parts[1].strip() if len(parts) > 1 else "వివరాలు నమోదు చేయబడ్డాయి."
                return {"english": english_part, "telugu": telugu_part}
            return {"english": transcript_text, "telugu": transcript_text}