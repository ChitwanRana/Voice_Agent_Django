import logging
import json
import httpx
import base64
import io
import azure.cognitiveservices.speech as speechsdk
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

# Cache Speech SDK components globally
_speech_config = None
_speech_synthesizer = None

def get_speech_synthesizer():
    """Initialize and cache Speech SDK synthesizer"""
    global _speech_config, _speech_synthesizer
    
    if _speech_synthesizer is None:
        logger.debug("Initializing Speech SDK synthesizer")
        config = MyConfig.envFile()
        
        _speech_config = speechsdk.SpeechConfig(
            subscription=config["SPEECH_KEY"],
            region=config["SPEECH_REGION"]
        )
        
        # Set to Hindi language for synthesis
        _speech_config.speech_recognition_language = "hi-IN"
        # Use native Hindi voice
        _speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
        
        # Create synthesizer without audio output (we'll get the audio data)
        _speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=_speech_config,
            audio_config=None  # No audio output, we'll capture the data
        )
        
        logger.info("Speech SDK synthesizer initialized with Hindi voice: hi-IN-SwaraNeural")
    
    return _speech_synthesizer


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

        # Enhanced system prompt for Hindi responses
        base_personality = """आप एक सहायक AI वॉइस असिस्टेंट हैं।
- उपयोगकर्ता के प्रश्न का सीधा, संक्षिप्त उत्तर दें
- वॉइस इंटरैक्शन के लिए अधिकतम 1-2 वाक्यों में जवाब दें
- उपयोगकर्ता की भाषा से मेल खाएं - यदि वे हिंगलिश बोलते हैं, तो हिंगलिश में जवाब दें
- बिना अभिवादन, बिना अतिरिक्त स्पष्टीकरण (जब तक नहीं पूछा जाए)
- स्वाभाविक और संवादात्मक रहें लेकिन संक्षिप्त रहें
- सीधे प्रश्न का उत्तर दें"""

        # Prepare system prompt
        if selected_domain == "normal":
            system_prompt = base_personality
        else:
            kb_text = load_kb(selected_domain)
            system_prompt = (
                f"{base_personality}\n\n"
                f"केवल {selected_domain} नॉलेज बेस का उपयोग करके उत्तर दें।\n"
                f"विशिष्ट जानकारी के साथ सीधे उत्तर दें (डॉक्टर के नाम, कमरा संख्या, समय)।\n"
                f"यदि जानकारी उपलब्ध नहीं है, तो कहें: 'क्षमा करें, मेरे पास यह जानकारी नहीं है।'\n\n"
                f"--- KB START ---\n{kb_text}\n--- KB END ---"
            )
        
        # Force Hindi responses if Devanagari detected
        if is_hindi_script:
            system_prompt = f"हिंदी में उत्तर दें। {system_prompt}"

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
                    max_tokens=150,
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
    """Text-to-Speech using Azure Speech SDK ONLY - native Hindi voice"""
    logger.info("api_tts called. method=%s remote=%s", request.method, request.META.get("REMOTE_ADDR"))
    if request.method != "POST":
        logger.warning("api_tts: non-POST request")
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body)
        text = (payload.get("text") or "").strip()
        logger.debug("api_tts: text_len=%d text_preview=%s", len(text), text[:100])

        if not text:
            logger.warning("api_tts: empty text")
            return JsonResponse({"error": "Empty text"}, status=400)

        # Detect language
        is_hindi = _contains_devanagari(text)
        
        # Get Speech SDK synthesizer
        synthesizer = get_speech_synthesizer()
        
        # Choose voice based on content
        if is_hindi:
            voice_name = "hi-IN-SwaraNeural"  # Native Hindi female
            lang = "hi-IN"
        else:
            voice_name = "en-IN-NeerjaNeural"  # Indian English female
            lang = "en-IN"
        
        logger.debug("api_tts: using voice=%s lang=%s", voice_name, lang)

        # Create SSML for better pronunciation
        ssml = f"""<speak version='1.0' xml:lang='{lang}' xmlns='http://www.w3.org/2001/10/synthesis'>
            <voice name='{voice_name}'>
                <prosody rate='0.95' pitch='0%'>
                    {text}
                </prosody>
            </voice>
        </speak>"""

        try:
            # Synthesize speech using Azure SDK
            logger.info("api_tts: starting synthesis with Azure Speech SDK")
            result = synthesizer.speak_ssml_async(ssml).get()
            
            # Check synthesis result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Get audio data
                audio_data = result.audio_data
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                logger.info("api_tts: synthesis successful, audio_bytes=%d voice=%s", len(audio_data), voice_name)
                return JsonResponse({
                    "audio": audio_base64,
                    "format": "wav"  # Azure SDK returns WAV format by default
                })
                
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                error_msg = f"Speech synthesis canceled: {cancellation.reason}"
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    error_msg += f" - {cancellation.error_details}"
                logger.error("api_tts: %s", error_msg)
                return JsonResponse({"error": error_msg}, status=500)
            else:
                logger.error("api_tts: unexpected result reason: %s", result.reason)
                return JsonResponse({"error": "Speech synthesis failed"}, status=500)
                
        except Exception as e:
            logger.exception("api_tts: Azure Speech SDK error")
            return JsonResponse({"error": f"TTS error: {str(e)}"}, status=500)

    except Exception as e:
        logger.exception("api_tts: unexpected exception")
        return JsonResponse({"error": "TTS error"}, status=500)
