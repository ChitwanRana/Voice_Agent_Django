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
import html  

logger = logging.getLogger("voice_app")

def index(request):
    logger.debug("Rendering index page.")
    return render(request, "voice_app/index.html")

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
KB_FILES = {"healthcare": KB_DIR / "healthcare.md", "finance": KB_DIR / "finance.md"}

_azure_client = None
_speech_synthesizer = None

def get_azure_client():
    global _azure_client
    if _azure_client is None:
        logger.info("Initializing AzureOpenAI client for the first time.")
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
        logger.info("Initializing Azure SpeechSynthesizer for the first time.")
        config = MyConfig.envFile()
        speech_config = speechsdk.SpeechConfig(
            subscription=config["SPEECH_KEY"],
            region=config["SPEECH_REGION"]
        )
        speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
        # Use mp3 format for faster transfer and smaller payload
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3
        )
        _speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None
        )
    return _speech_synthesizer

def load_kb(domain: str) -> str:
    """Loads knowledge base content for a given domain, using cache if available."""
    if not domain or domain not in KB_FILES:
        return ""
    
    cache_key = f"kb_{domain}"
    if kb := cache.get(cache_key):
        logger.info(f"Knowledge base for domain '{domain}' found in cache.")
        return kb
    
    fp = KB_FILES.get(domain)
    if not fp or not fp.exists():
        logger.error(f"KB file for '{domain}' not found: {fp}")
        return ""
    
    try:
        logger.info(f"Loading knowledge base for domain '{domain}' from file: {fp}")
        content = fp.read_text("utf8").strip()
        if content:
            cache.set(cache_key, content, timeout=3600)
            logger.info(f"Knowledge base for domain '{domain}' cached successfully. Size: {len(content)} bytes")
        return content
    except Exception as e:
        logger.error(f"Failed to read KB for '{domain}': {e}")
        return ""

@csrf_exempt
def api_ask(request):
    """
    Handles user queries by streaming responses from Azure OpenAI, with enhanced
    context-awareness by leveraging chat history.
    """
    if request.method != "POST":
        logger.warning(f"Received {request.method} request for api_ask, but only POST is allowed.")
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body)
        user_text = (payload.get("text") or "").strip()
        if not user_text:
            return JsonResponse({"error": "Empty text"}, status=400)
        if len(user_text) > 1000:
            return JsonResponse({"error": "Text too long. Maximum 1000 characters allowed."}, status=400)

        selected_domain = (payload.get("domain") or "normal").strip().lower()
        if selected_domain in ("healthcare", "finance", "normal"):
            request.session["selected_domain"] = selected_domain
        else:
            selected_domain = request.session.get("selected_domain", "normal")

        # Log the user query
        logger.info(f"Processing user query in domain: '{selected_domain}'. User text: '{user_text[:50]}{'...' if len(user_text) > 50 else ''}'")

        history = request.session.get("chat_history", [])
        if len(history) > 20:
            history = history[-20:]
        history.append({"role": "user", "content": user_text})

        conversation_summary = "\n".join(
            f"- {msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}"
            for msg in history[-5:]
        )

        base_personality = f"""You are Bodhita AI, a friendly female voice assistant. This is a continuous conversation.

CURRENT CONTEXT:
- This is a voice-based chat. Keep responses concise (2-3 sentences).
- Your response MUST be a direct continuation of the previous conversation.
- ALWAYS respond in HINGLISH (Roman script). NEVER use Devanagari.
- You are a helpful, warm, and empathetic female assistant.
- Use a feminine, friendly conversational style in your responses.
- Recent conversation turns:
{conversation_summary}

YOUR TASK:
1. Acknowledge the user's latest message in the context of the history.
2. If the user is asking a follow-up question, connect your answer to what was discussed. Use phrases like "Jaise hum baat kar rahe the..." or "Uske baare mein aur batane ke liye...".
3. If the user changes the topic, acknowledge it and answer the new question.
4. Provide accurate, factual information in a friendly, warm, and conversational feminine tone.
5. Express empathy and understanding when appropriate.
"""

        system_prompt = base_personality
        if selected_domain != "normal":
            if not (kb_text := load_kb(selected_domain)):
                return JsonResponse({"error": f"Knowledge base for {selected_domain} is not available"}, status=503)
            system_prompt += f"""\n\nKNOWLEDGE BASE INSTRUCTIONS:\n- You are in {selected_domain.upper()} mode.\n- You MUST answer ONLY using the Knowledge Base below.\n- If the answer is not in the knowledge base, state that clearly, e.g., "Yeh jaankari mere pass nahi hai."\n- Refer to the conversation history for context, but derive your answer from the knowledge base.\n\n--- {selected_domain.upper()} KNOWLEDGE BASE ---\n{kb_text}\n---\n"""

        messages = [{"role": "system", "content": system_prompt}] + history

        collected_response = {"text": "", "error": None}

        def generate_stream():
            try:
                logger.info(f"Requesting chat completion with context. History length: {len(history)} messages.")
                stream = get_azure_client().chat.completions.create(
                    model=MyConfig.envFile()["AZURE_OPENAI_DEPLOYMENT_NAME"],
                    messages=messages,
                    max_tokens=150,
                    temperature=0.3,
                    stream=True
                )

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        collected_response["text"] += content
                        yield f"data: {json.dumps({'chunk': content})}\n\n"

                logger.info(f"Finished streaming response. Total length: {len(collected_response['text'])} chars")
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                logger.error(f"Stream error: {e}", exc_info=True)
                collected_response["error"] = str(e)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        def stream_with_session_update():
            yield from generate_stream()
            if collected_response["text"] and not collected_response["error"]:
                history.append({"role": "assistant", "content": collected_response["text"]})
                request.session["chat_history"] = history
                request.session.modified = True
                logger.info(f"Updated chat history. New length: {len(history)} messages.")

        return StreamingHttpResponse(
            stream_with_session_update(),
            content_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {e}")
        return JsonResponse({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        logger.error(f"Unexpected error in api_ask view: {e}", exc_info=True)
        return JsonResponse({"error": "Internal server error"}, status=500)

@csrf_exempt
def reset_context(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    
    try:
        request.session["chat_history"] = []
        request.session.pop("selected_domain", None)
        request.session.modified = True
        logger.info("Conversation history cleared successfully.")
        return JsonResponse({"status": "success", "message": "Conversation history cleared"})
    except Exception as e:
        logger.error(f"Reset error: {e}")
        return JsonResponse({"error": "Failed to reset context"}, status=500)

@csrf_exempt
def get_chat_history(request):
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)
    
    history = request.session.get("chat_history", [])
    logger.info(f"Fetching chat history. Count: {len(history)} messages.")
    return JsonResponse({
        "history": history,
        "count": len(history),
        "domain": request.session.get("selected_domain", "normal")
    })

@csrf_exempt
def api_tts(request):
    """Handles text-to-speech conversion using Azure Speech Service."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body)
        text = (payload.get("text") or "").strip()
        if not text:
            return JsonResponse({"error": "Empty text"}, status=400)
        if len(text) > 5000:
            return JsonResponse({"error": "Text too long for synthesis. Maximum 5000 characters."}, status=400)

        logger.info(f"Requesting TTS for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        # Check cache first for faster repeated responses
        cache_key = f"tts_{hash(text)}"
        if cached_audio := cache.get(cache_key):
            logger.info("Returning cached TTS audio.")
            return JsonResponse(cached_audio)

        # Optimized SSML with faster speech rate for quicker playback
        ssml = f"""<speak version='1.0' xml:lang='hi-IN'><voice name='hi-IN-SwaraNeural'><prosody rate='1.2'>{html.escape(text, quote=False)}</prosody></voice></speak>"""
        logger.debug(f"Generated SSML: {ssml[:200]}...")

        # Use the cached synthesizer for faster response
        result = get_speech_synthesizer().speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Collect all audio data
            audio_data = bytes(result.audio_data)
            
            if not audio_data:
                logger.error("No audio data generated by TTS synthesis.")
                return JsonResponse({"error": "No audio data generated"}, status=500)
            
            logger.info(f"Successfully synthesized audio. Size: {len(audio_data)} bytes")
            
            # Return direct JSON response with base64 encoded audio
            response_data = {
                "audio": base64.b64encode(audio_data).decode('utf-8'),
                "format": "mp3"
            }
            
            # Cache the complete audio for future requests
            cache.set(cache_key, response_data, timeout=3600)
            
            return JsonResponse(response_data)
        
        error_msg = f"Reason: {result.cancellation_details.reason}, Details: {result.cancellation_details.error_details}"
        logger.error(f"TTS synthesis failed. {error_msg}")
        return JsonResponse({"error": "Speech synthesis failed", "details": error_msg}, status=500)

    except json.JSONDecodeError:
        logger.error("Invalid JSON in TTS request body.")
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        return JsonResponse({"error": "Internal server error"}, status=500)
