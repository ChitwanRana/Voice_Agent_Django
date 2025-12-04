import logging
import json
import httpx
import base64
from django.core.files.base import ContentFile
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from openai import AzureOpenAI
from django.core.cache import cache
from src.config.config import MyConfig
from src.prompts.system_prompt import VOICE_ASSISTANT_PROMPT
from pathlib import Path

logger = logging.getLogger("voice_app")
 
def index(request):
    """Serve the SPA page."""
    logger.info("index view called. method=%s remote=%s", request.method, request.META.get("REMOTE_ADDR"))
    return render(request, "voice_app/index.html")

# Ensure logger prints useful info to console during development if not configured by Django settings
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

MAX_HISTORY = 10

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
KB_FILES = {
    "healthcare": KB_DIR / "healthcare.md",
    "finance": KB_DIR / "finance.md",
}

# Cache the Azure client globally
_azure_client = None

def get_azure_client():
    global _azure_client
    if _azure_client is None:
        logger.debug("get_azure_client: initializing Azure client")
        try:
            config = MyConfig.envFile()
            logger.debug("get_azure_client: loaded config keys: %s", ", ".join([k for k in config.keys() if k]))
            _azure_client = AzureOpenAI(
                api_key=config["AZURE_OPENAI_KEY"],
                api_version=config["AZURE_OPENAI_API_VERSION"],
                azure_endpoint=config["AZURE_OPENAI_ENDPOINT"],
                http_client=httpx.Client(
                    timeout=10,
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
                )
            )
            logger.info("get_azure_client: Azure client initialized")
        except Exception:
            logger.exception("get_azure_client: failed to initialize Azure client")
            raise
    return _azure_client


def load_kb(domain):
    logger.debug("load_kb: domain=%s", domain)
    kb = cache.get(domain)
    if kb:
        logger.debug("load_kb: cache hit for domain=%s (size=%d)", domain, len(kb))
        return kb
    fp = KB_FILES.get(domain)
    if not fp:
        logger.warning("load_kb: no KB file mapped for domain=%s", domain)
        return ""
    try:
        content = fp.read_text("utf8")
        cache.set(domain, content, timeout=None)
        logger.info("load_kb: loaded KB file %s (bytes=%d)", fp, len(content.encode("utf8")))
        return content
    except Exception:
        logger.exception("load_kb: failed reading KB file %s", fp)
        return ""


def _contains_devanagari(text: str) -> bool:
    if not text:
        return False
    return any('\u0900' <= ch <= '\u097F' for ch in text)

@csrf_exempt
def api_ask(request):
    logger.info("api_ask called. method=%s remote=%s", request.method, request.META.get("REMOTE_ADDR"))
    if request.method != "POST":
        logger.warning("api_ask: non-POST request")
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body)
        logger.debug("api_ask: payload keys=%s body_size=%d", list(payload.keys()), len(request.body or b""))
        user_text = (payload.get("text") or "").strip()
        is_hindi_script = _contains_devanagari(user_text)
        logger.debug("api_ask: user_text_len=%d is_hindi_script=%s", len(user_text), is_hindi_script)

        selected_domain = (payload.get("domain") or "").strip().lower()
        if selected_domain in ("healthcare", "finance", "normal"):
            request.session["selected_domain"] = selected_domain
            logger.debug("api_ask: domain set from payload=%s", selected_domain)
        else:
            selected_domain = request.session.get("selected_domain", "normal")
            logger.debug("api_ask: domain from session or default=%s", selected_domain)

        if not user_text:
            logger.warning("api_ask: empty user_text")
            return JsonResponse({"error": "Empty text"}, status=400)

        history = request.session.get("chat_history", [])
        logger.debug("api_ask: prior history length=%d", len(history))
        history.append({"role": "user", "content": user_text})

        if len(history) > 6:
            history = history[-6:]
            logger.debug("api_ask: trimmed history to last 6 entries")

        # Enhanced system prompt for direct, concise answers
        base_personality = """You are a helpful AI voice assistant. 
- Give DIRECT, SHORT answers to what the user asks
- Answer in 1-2 sentences maximum for voice interaction
- Match the user's language - if they speak Hinglish, reply in Hinglish
- NO greetings, NO extra explanations unless asked
- Be natural and conversational but BRIEF
- Just answer the question directly"""

        # Prepare system prompt
        if selected_domain == "normal":
            system_prompt = base_personality
        else:
            kb_text = load_kb(selected_domain)
            system_prompt = (
                f"{base_personality}\n\n"
                f"Answer ONLY using the {selected_domain} knowledge base below.\n"
                f"Give direct answers with specific information (doctor names, room numbers, timings).\n"
                f"If information is missing, say: 'Sorry, I don't have that information.'\n\n"
                f"--- KB START ---\n{kb_text}\n--- KB END ---"
            )
        if is_hindi_script:
            system_prompt = f"Reply in Hindi. {system_prompt}"

        logger.debug("api_ask: system_prompt_len=%d preview=%s", len(system_prompt), system_prompt[:300].replace("\n", " "))

        messages = [{"role": "system", "content": system_prompt}] + history
        logger.debug("api_ask: messages_count=%d last_user_preview=%s", len(messages), user_text[:120])

        # Stream generator
        def generate_stream():
            client = get_azure_client()
            full_response = ""
            logger.info("generate_stream: starting stream for model=%s", MyConfig.envFile().get("AZURE_OPENAI_DEPLOYMENT_NAME"))
            try:
                stream = client.chat.completions.create(
                    model=MyConfig.envFile()["AZURE_OPENAI_DEPLOYMENT_NAME"],
                    messages=messages,
                    max_tokens=120,
                    temperature=0.7,
                    stream=True
                )

                for chunk in stream:
                    try:
                        if (hasattr(chunk, 'choices') and 
                            len(chunk.choices) > 0 and 
                            hasattr(chunk.choices[0], 'delta') and 
                            hasattr(chunk.choices[0].delta, 'content') and
                            chunk.choices[0].delta.content):
                            
                            content = chunk.choices[0].delta.content
                            logger.debug("generate_stream: received chunk len=%d snippet=%s", len(content), content[:120].replace("\n", " "))
                            full_response += content
                            yield f"data: {json.dumps({'chunk': content})}\n\n"
                    except Exception:
                        logger.exception("generate_stream: error processing chunk, continuing")

                logger.info("generate_stream: stream finished, total_response_len=%d", len(full_response))
                yield f"data: {json.dumps({'done': True})}\n\n"

                if full_response:
                    history.append({"role": "assistant", "content": full_response})
                    request.session["chat_history"] = history
                    request.session.modified = True
                    response_lang = "hi" if _contains_devanagari(full_response) else ""
                    request.session["last_response_lang"] = response_lang
                    logger.debug("generate_stream: stored assistant response, lang=%s", response_lang)

            except Exception as e:
                logger.exception("generate_stream: exception while streaming")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingHttpResponse(
            generate_stream(),
            content_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
            }
        )

    except Exception as e:
        logger.exception("api_ask: unexpected exception")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def reset_context(request):
    logger.info("reset_context endpoint hit. Method: %s", request.method)

    if request.method != "POST":
        logger.warning("Invalid method used on reset_context: %s", request.method)
        return JsonResponse({"error": "POST required."}, status=405)

    request.session["chat_history"] = []
    request.session.modified = True
    logger.info("Chat context successfully reset.")

    return JsonResponse({"status": "context reset"})

@csrf_exempt
def api_tts(request):
    """Text-to-Speech using Azure Speech SDK with Arjuna/Madhur multilingual voice (Indian accent)"""
    logger.info("api_tts called. method=%s remote=%s", request.method, request.META.get("REMOTE_ADDR"))
    if request.method != "POST":
        logger.warning("api_tts: non-POST request")
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body)
        text = (payload.get("text") or "").strip()
        lang = (payload.get("lang") or "").strip().lower()  # 'hi' or 'en'
        logger.debug("api_tts: text_len=%d lang=%s", len(text), lang)

        if not text:
            logger.warning("api_tts: empty text")
            return JsonResponse({"error": "Empty text"}, status=400)

        config = MyConfig.envFile()
        speech_key = config.get("SPEECH_KEY") or config.get("AZURE_SPEECH_KEY")
        service_region = config.get("SPEECH_REGION") or config.get("AZURE_SPEECH_REGION")

        if not speech_key or not service_region:
            logger.error("api_tts: missing speech config. keys_present=%s region=%s", bool(speech_key), service_region)
            return JsonResponse({"error": "Speech config missing"}, status=500)

        # Use Arjuna/Madhur multilingual voice for Indian accent (both Hindi and English)
        # Default to male multilingual voice
        chosen_voice = config.get("AZURE_TTS_VOICE", "hi-IN-MadhurMultilingualNeural")
        
        # Alternative voices:
        # "hi-IN-SwaraMultilingualNeural" - Female multilingual (Hindi + English)
        # "hi-IN-MadhurMultilingualNeural" - Male multilingual (Hindi + English) - Arjuna
        
        # Set language for proper pronunciation
        xml_lang = "hi-IN" if lang == "hi" else "en-IN"
        
        endpoint = f"https://{service_region}.tts.speech.microsoft.com/cognitiveservices/v1"

        logger.debug("api_tts: endpoint=%s chosen_voice=%s xml_lang=%s", endpoint, chosen_voice, xml_lang)

        headers = {
            "Ocp-Apim-Subscription-Key": speech_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3"
        }

        # SSML with multilingual voice - handles both Hindi and English with Indian accent
        ssml = f"""<speak version='1.0' xml:lang='{xml_lang}'>
            <voice xml:lang='{xml_lang}' name='{chosen_voice}'>
                <prosody rate='1.0' pitch='0%'>
                    {text}
                </prosody>
            </voice>
        </speak>"""

        try:
            response = httpx.post(endpoint, headers=headers, content=ssml, timeout=30)
            logger.debug("api_tts: httpx.post returned status=%s bytes=%d", response.status_code, len(response.content or b""))
            if response.status_code == 200:
                audio_base64 = base64.b64encode(response.content).decode('utf-8')
                logger.info("api_tts: TTS succeeded, audio_bytes=%d voice=%s", len(response.content), chosen_voice)
                return JsonResponse({"audio": audio_base64, "format": "mp3"})
            else:
                truncated = (response.text or "")[:1000]
                logger.error("api_tts: TTS API error: %s - %s", response.status_code, truncated)
                return JsonResponse({"error": f"TTS failed: {response.status_code}"}, status=500)
        except httpx.RequestError:
            logger.exception("api_tts: httpx request error when calling TTS endpoint")
            return JsonResponse({"error": "TTS request failed"}, status=500)

    except Exception:
        logger.exception("api_tts: unexpected exception")
        return JsonResponse({"error": "TTS error"}, status=500)
