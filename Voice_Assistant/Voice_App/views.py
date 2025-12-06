import logging
import json
import base64
import azure.cognitiveservices.speech as speechsdk          #type: ignore
from django.shortcuts import render                         #type: ignore
from django.http import JsonResponse, StreamingHttpResponse #type: ignore
from django.views.decorators.csrf import csrf_exempt        #type: ignore
from openai import AzureOpenAI                              #type: ignore
from django.core.cache import cache                         #type: ignore
from src.config.config import MyConfig
from pathlib import Path

logger = logging.getLogger("voice_app")

def index(request):
    return render(request, "voice_app/index.html")

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
KB_FILES = {"healthcare": KB_DIR / "healthcare.md", "finance": KB_DIR / "finance.md"}

_azure_client = None
_speech_synthesizer = None

def get_azure_client():
    global _azure_client
    if _azure_client is None:
        config = MyConfig.envFile()
        _azure_client = AzureOpenAI(
            api_key=config["AZURE_OPENAI_KEY"],
            api_version=config["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=config["AZURE_OPENAI_ENDPOINT"]
        )
    return _azure_client

def get_speech_synthesizer():
    global _speech_synthesizer
    if _speech_synthesizer is None:
        config = MyConfig.envFile()
        speech_config = speechsdk.SpeechConfig(
            subscription=config["SPEECH_KEY"],
            region=config["SPEECH_REGION"]
        )
        speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
        _speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None
        )
    return _speech_synthesizer

def load_kb(domain):
    kb = cache.get(domain)
    if kb:
        return kb
    fp = KB_FILES.get(domain)
    if not fp or not fp.exists():
        return ""
    try:
        content = fp.read_text("utf8")
        cache.set(domain, content, timeout=None)
        return content
    except Exception:
        return ""

def _contains_devanagari(text: str) -> bool:
    return any('\u0900' <= ch <= '\u097F' for ch in text) if text else False

@csrf_exempt
def api_ask(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body)
        user_text = (payload.get("text") or "").strip()
        if not user_text:
            return JsonResponse({"error": "Empty text"}, status=400)

        selected_domain = (payload.get("domain") or "").strip().lower()
        if selected_domain in ("healthcare", "finance", "normal"):
            request.session["selected_domain"] = selected_domain
        else:
            selected_domain = request.session.get("selected_domain", "normal")

        history = request.session.get("chat_history", [])
        history.append({"role": "user", "content": user_text})
        if len(history) > 6:
            history = history[-6:]

        # Improved system prompt for informative, knowledge-focused responses
        base_personality = """You are Bodhita AI, an expert voice assistant specializing in providing accurate, informative answers.

RESPONSE STYLE:
- Always respond in HINGLISH (Hindi-English mix using Roman script ONLY)
- Be informative and educational - focus on facts, explanations, and practical information
- Keep answers concise but comprehensive (2-3 sentences for voice)
- Use simple language that's easy to understand
- Include key facts, numbers, or important details when relevant
- Sound professional yet friendly and conversational

LANGUAGE FORMAT:
- Use Hindi words in Roman script mixed with English
- Example: "Diabetes ek metabolic disorder hai jisme blood glucose level abnormally high ho jata hai"
- NEVER use Devanagari (देवनागरी) script
- Natural code-mixing between Hindi and English

INFORMATION PRIORITY:
1. Provide accurate, factual information first
2. Explain the "what" and "why" clearly
3. Add practical tips or implications if relevant
4. Keep it conversational for voice interaction

EXAMPLES:

User: "What is high blood pressure?"
You: "High blood pressure ya hypertension ek condition hai jisme aapki arteries mein blood ka pressure consistently 140/90 mmHg se zyada rehta hai. Isse heart attack, stroke aur kidney problems ka risk badh jata hai, isliye regular monitoring aur healthy lifestyle bahut zaroori hai."

User: "How do credit cards work?"
You: "Credit card ek short-term loan hai jo bank aapko deta hai. Aap items purchase karte hain aur bank vendor ko pay karta hai, phir aapko wo amount interest-free period mein wapas karna hota hai. Agar time pe payment nahi hui toh 24-42% annual interest charge hota hai."

User: "What are symptoms of diabetes?"
You: "Diabetes ke main symptoms hain - excessive thirst aur hunger, frequent urination, unexplained weight loss, aur fatigue. Wounds slowly heal hote hain aur vision bhi blurry ho sakti hai. Agar ye symptoms dikhein toh immediately doctor se consult karein."
"""

        if selected_domain == "normal":
            system_prompt = base_personality
        else:
            kb_text = load_kb(selected_domain)
            system_prompt = f"""{base_personality}

KNOWLEDGE BASE INSTRUCTIONS:
- You are now in {selected_domain.upper()} domain mode
- Use ONLY the information from the knowledge base provided below
- If asked something not in the knowledge base, politely say "Ye information mere knowledge base mein available nahi hai"
- Cite specific facts, numbers, and details from the knowledge base
- Be authoritative and accurate based on the provided information

--- {selected_domain.upper()} KNOWLEDGE BASE ---
{kb_text}
---
"""

        messages = [{"role": "system", "content": system_prompt}] + history

        def generate_stream():
            client = get_azure_client()
            full_response = ""
            try:
                stream = client.chat.completions.create(
                    model=MyConfig.envFile()["AZURE_OPENAI_DEPLOYMENT_NAME"],
                    messages=messages,
                    max_tokens=200,  # Increased for more informative responses
                    temperature=0.6,  # Lowered for more factual, consistent responses
                    stream=True
                )

                for chunk in stream:
                    try:
                        if (hasattr(chunk, 'choices') and len(chunk.choices) > 0 and 
                            hasattr(chunk.choices[0], 'delta') and 
                            hasattr(chunk.choices[0].delta, 'content') and
                            chunk.choices[0].delta.content):
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield f"data: {json.dumps({'chunk': content})}\n\n"
                    except Exception:
                        continue

                yield f"data: {json.dumps({'done': True})}\n\n"

                if full_response:
                    history.append({"role": "assistant", "content": full_response})
                    request.session["chat_history"] = history
                    request.session.modified = True

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingHttpResponse(
            generate_stream(),
            content_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def reset_context(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required."}, status=405)
    request.session["chat_history"] = []
    request.session.modified = True
    return JsonResponse({"status": "context reset"})

@csrf_exempt
def api_tts(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body)
        text = (payload.get("text") or "").strip()
        if not text:
            return JsonResponse({"error": "Empty text"}, status=400)

        synthesizer = get_speech_synthesizer()
        is_hindi = _contains_devanagari(text)
        voice_name = "hi-IN-SwaraNeural" if is_hindi else "en-IN-NeerjaNeural"
        lang = "hi-IN" if is_hindi else "en-IN"

        ssml = f"""<speak version='1.0' xml:lang='{lang}'>
            <voice name='{voice_name}'>
                <prosody rate='1.1' pitch='0%'>{text}</prosody>
            </voice>
        </speak>"""

        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_base64 = base64.b64encode(result.audio_data).decode('utf-8')
            return JsonResponse({"audio": audio_base64, "format": "wav"})
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            return JsonResponse({"error": f"Synthesis canceled: {cancellation.reason}"}, status=500)
        else:
            return JsonResponse({"error": "Synthesis failed"}, status=500)

    except Exception as e:
        return JsonResponse({"error": f"TTS error: {str(e)}"}, status=500)
