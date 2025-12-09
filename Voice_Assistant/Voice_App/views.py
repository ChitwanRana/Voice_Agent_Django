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

        # Enhanced chat history management
        history = request.session.get("chat_history", [])
        history.append({"role": "user", "content": user_text})
        
        # Keep last 10 messages (5 exchanges) for better context
        if len(history) > 10:
            history = history[-10:]

        # Improved system prompt with memory awareness
        base_personality = """You are Bodhita AI, an expert voice assistant specializing in providing accurate, informative answers.

MEMORY & CONTEXT:
- You have access to the conversation history
- Reference previous questions/answers when relevant
- Use phrases like "Jaise maine pehle bataya tha..." or "Aapne jo pehle pucha tha uske baare mein..."
- Connect current questions to past context naturally
- If user asks follow-up questions, understand the context from history

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
1. Check conversation history for context
2. Provide accurate, factual information
3. Explain the "what" and "why" clearly
4. Add practical tips or implications if relevant
5. Keep it conversational for voice interaction

EXAMPLES WITH MEMORY:

Conversation 1:
User: "What is diabetes?"
You: "Diabetes ek metabolic disorder hai jisme aapke blood sugar level abnormally high ho jata hai. Ye insulin production ya insulin resistance ki wajah se hota hai, aur agar control na kiya jaye toh serious complications ho sakti hain."

User: "What are its symptoms?"
You: "Diabetes ke main symptoms hain - excessive thirst aur frequent urination, unexplained weight loss, fatigue, aur blurred vision. Wounds bhi slowly heal hote hain. Agar ye symptoms dikhein toh turant doctor se consult karein."

User: "How to control it?"
You: "Diabetes ko control karne ke liye balanced diet lein jisme low sugar aur high fiber ho, regular exercise karein minimum 30 minutes daily, aur doctor ki prescribed medicines time pe lein. Regular blood sugar monitoring bhi bahut important hai."

Conversation 2:
User: "Tell me about loans"
You: "Loan ek borrowed money hai jo aap bank ya financial institution se lete hain aur fixed tenure mein interest ke saath repay karte hain. Home loan, personal loan, car loan jaise different types available hain, aur interest rates 8-15% tak hote hain."

User: "Which one is best?"
You: "Ye aapki requirement pe depend karta hai. Home loan sabse low interest rate pe milta hai around 8-9%, aur tax benefits bhi hain. Personal loan quickly mil jata hai but interest high hota hai 12-15%. Apni priority aur repayment capacity dekh ke decide karein."
"""

        if selected_domain == "normal":
            system_prompt = base_personality
        else:
            kb_text = load_kb(selected_domain)
            system_prompt = f"""{base_personality}

KNOWLEDGE BASE INSTRUCTIONS:
- You are now in {selected_domain.upper()} domain mode
- Use ONLY the information from the knowledge base provided below
- Reference conversation history for context but answer from knowledge base
- If asked something not in the knowledge base, politely say "Ye specific information mere knowledge base mein available nahi hai, lekin jo aapne pehle pucha tha uske baare mein main bata sakta hoon"
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
                    max_tokens=200,  # Increased for more detailed contextual responses
                    temperature=0.7,  # Slightly higher for more natural conversational flow
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
                    
                    # Log conversation for debugging
                    logger.info(f"Chat history length: {len(history)} messages")

            except Exception as e:
                logger.error(f"Stream error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingHttpResponse(
            generate_stream(),
            content_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

    except Exception as e:
        logger.error(f"API ask error: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def reset_context(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required."}, status=405)
    
    # Clear chat history
    request.session["chat_history"] = []
    request.session.modified = True
    
    logger.info("Chat history reset")
    return JsonResponse({"status": "context reset", "message": "Conversation history cleared"})

@csrf_exempt
def get_chat_history(request):
    """New endpoint to retrieve current chat history"""
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)
    
    history = request.session.get("chat_history", [])
    return JsonResponse({
        "history": history,
        "count": len(history),
        "domain": request.session.get("selected_domain", "normal")
    })

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
            logger.error(f"TTS canceled: {cancellation.reason}")
            return JsonResponse({"error": f"Synthesis canceled: {cancellation.reason}"}, status=500)
        else:
            return JsonResponse({"error": "Synthesis failed"}, status=500)

    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        return JsonResponse({"error": f"TTS error: {str(e)}"}, status=500)
