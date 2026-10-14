import os
import re
import logging
from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types

logger = logging.getLogger("state_machine")
logging.basicConfig(level=logging.INFO)

# Use the active alias discovered from your API key that does not suffer from 503 spikes
TARGET_MODEL = "gemini-flash-latest"

class CallStateMachine:
    def __init__(self, caller_number: str, contact_name: Optional[str] = None):
        self.caller_number = caller_number
        self.contact_name = contact_name.strip() if contact_name and contact_name.strip() else None
        self.locked_language: Optional[str] = None  # 'te', 'hi', or 'en'
        self.history: List[Dict[str, str]] = []
        self.turn_count = 0
        self.is_completed = False

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.client = genai.Client(api_key=api_key) if api_key else None

    def get_initial_greeting(self) -> str:
        """Rule 1a: Polite trilingual greeting."""
        return "Hello, Namaste, Namaskaram! I am an automated assistant answering on behalf of the user. How can I help you today?"

    def detect_and_lock_language(self, caller_text: str) -> str:
        """Rule 1b & 1c: Detect language on Turn 1 and lock permanently."""
        # Telugu Unicode or phonetic romanized cues
        has_telugu_script = bool(re.search(r'[\u0C00-\u0C7F]', caller_text))
        telugu_keywords = ["namaskaram", "bagunnara", "cheppandi", "telugu", "enti", "kavalandi", "matladali", "arjent", "urgent"]
        is_telugu = has_telugu_script or any(w in caller_text.lower() for w in telugu_keywords)

        # Devanagari Unicode or phonetic Hindi cues
        has_hindi_script = bool(re.search(r'[\u0900-\u097F]', caller_text))
        hindi_keywords = ["namaste", "kaise ho", "bataye", "hindi", "kya", "bhai", "haan", "baat", "zaroori"]
        is_hindi = has_hindi_script or any(w in caller_text.lower() for w in hindi_keywords)

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
            f"The caller's identity is verified from contacts as '{self.contact_name}'. Greet them politely by name. Do NOT ask for their name."
            if self.contact_name
            else "The caller's name is UNKNOWN. Politely ask for their full name during the conversation."
        )

        return f"""
Purpose and Goals:
You are an automated personal phone call assistant answering incoming calls on behalf of the user.
Speak exclusively and fluently in {current_lang}.

Behaviors and Rules:
1. Language Lock: Transition immediately and maintain all subsequent conversation strictly in {current_lang}. Never switch languages.
2. Caller Identity: {name_instruction}
3. Conversational Cadence: Keep responses concise, clear, and direct. Strictly under 2 short sentences (maximum 25 words).
4. Direct Connection Guardrail: If asked to connect directly to the user or check availability, say:
   - Telugu: "వారు ప్రస్తుతం అందుబాటులో లేరు. నేను మీ ముఖ్యమైన సమాచారాన్ని నమోదు చేసి వెంటనే వారికి తెలియజేస్తాను."
   - Hindi: "वे अभी उपलब्ध नहीं हैं। मैं आपका संदेश नोट कर लेता हूँ और उन्हें तुरंत सूचित कर दूँगा।"
   - English: "The user is currently unavailable. I would be glad to take a detailed message and notify them immediately."
5. Gather Details: Inquire politely about the purpose of the call, caller's name (if not known), contact info, and degree of urgency (Urgent vs Standard).
6. Verification & Wrap-up:
   - Once details are gathered, summarize the key points back to the caller in {current_lang} to confirm accuracy.
   - Thank the caller in {current_lang} and conclude the call courteously (e.g., goodbye, ధన్యవాదాలు, धन्यवाद).
7. Format: Conversational spoken dialogue only. No asterisks, markdown, bullet points, or emojis.
"""

    def _generate_ai_reply(self, prompt: str, max_tokens: int = 80) -> Optional[str]:
        """Calls the discovered active gemini-flash-latest model."""
        if not self.client:
            logger.error("No Gemini Client initialized.")
            return None

        # Prioritize gemini-flash-latest followed by gemini-2.5-flash
        models_to_try = [TARGET_MODEL, "gemini-2.5-flash"]

        for model_name in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=max_tokens,
                    ),
                )
                if response.text:
                    return re.sub(r'[*_#`]', '', response.text).strip()
            except Exception as e:
                logger.warning(f"Generation attempt on {model_name} failed: {e}")
        return None

    async def process_turn(self, caller_text: str) -> str:
        self.turn_count += 1

        if not self.locked_language:
            self.detect_and_lock_language(caller_text)

        self.history.append({"role": "user", "text": caller_text})

        conversation_context = f"System Instructions:\n{self._build_system_prompt()}\n\n"
        for turn in self.history:
            speaker = "Caller" if turn["role"] == "user" else "Assistant"
            conversation_context += f"{speaker}: {turn['text']}\n"
        conversation_context += "Assistant:"

        clean_reply = self._generate_ai_reply(conversation_context, max_tokens=80)

        if clean_reply:
            self.history.append({"role": "model", "text": clean_reply})
            if any(w in clean_reply.lower() for w in ["goodbye", "ధన్యవాదాలు", "धन्यवाद", "thank you", "సెలవు"]):
                self.is_completed = True
            return clean_reply

        return self._fallback_reply()

    def _fallback_reply(self) -> str:
        if self.locked_language == "te":
            return "క్షమించండి, మీ మాటలు సరిగ్గా వినిపించలేదు. దయచేసి మళ్లీ చెప్పగలరా?"
        elif self.locked_language == "hi":
            return "माफ़ कीजिए, मैं सुन नहीं सका। क्या आप फिर से कह सकते हैं?"
        return "I apologize, could you please repeat that?"

    async def generate_bilingual_summary(self) -> Dict[str, str]:
        """Summarizes transcript strictly in English and Telugu."""
        transcript_text = "\n".join([
            f"{'Caller' if m['role'] == 'user' else 'AI'}: {m['text']}"
            for m in self.history
        ])

        summary_prompt = f"""
Summarize this call transcript into two distinct sections: English and Telugu.
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
        summary_result = self._generate_ai_reply(summary_prompt, max_tokens=300)

        if summary_result and "---TELUGU---" in summary_result:
            parts = summary_result.split("---TELUGU---")
            english_part = parts[0].replace("---ENGLISH---", "").strip()
            telugu_part = parts[1].strip()
            return {"english": english_part, "telugu": telugu_part}

        return {"english": transcript_text, "telugu": transcript_text}