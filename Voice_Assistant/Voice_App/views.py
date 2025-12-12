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
    """Renders the main chat page."""
    logger.debug("Rendering index page.")
    return render(request, "voice_app/index.html")

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
KB_FILES = {"healthcare": KB_DIR / "healthcare.md", "finance": KB_DIR / "finance.md"}

_azure_client = None
_speech_synthesizer = None

def get_azure_client():
    """Initializes and returns a singleton AzureOpenAI client."""
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
    """Initializes and returns a singleton SpeechSynthesizer client."""
    global _speech_synthesizer
    if _speech_synthesizer is None:
        logger.info("Initializing Azure SpeechSynthesizer for the first time.")
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

def load_kb(domain: str) -> str:
    """Loads knowledge base content for a given domain, using cache if available."""
    if not domain or domain not in KB_FILES:
        logger.warning(f"Invalid domain requested: '{domain}'")
        return ""
    
    # Try to get from cache first
    cache_key = f"kb_{domain}"
    kb = cache.get(cache_key)
    if kb:
        logger.info(f"Knowledge base for domain '{domain}' found in cache.")
        return kb
    
    fp = KB_FILES.get(domain)
    if not fp or not fp.exists():
        logger.error(f"Knowledge base file for domain '{domain}' not found at path: {fp}")
        return ""
    
    try:
        logger.info(f"Loading knowledge base for domain '{domain}' from file: {fp}")
        content = fp.read_text("utf8")
        
        if not content.strip():
            logger.warning(f"Knowledge base file for domain '{domain}' is empty")
            return ""
        
        # Cache for 1 hour (3600 seconds) instead of indefinitely
        cache.set(cache_key, content, timeout=3600)
        logger.info(f"Knowledge base for domain '{domain}' cached successfully. Size: {len(content)} bytes")
        return content
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading knowledge base file for domain '{domain}': {e}")
        return ""
    except Exception as e:
        logger.error(f"Failed to read knowledge base file for domain '{domain}': {e}", exc_info=True)
        return ""

def _contains_devanagari(text: str) -> bool:
    return any('\u0900' <= ch <= '\u097F' for ch in text) if text else False

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
            logger.warning("API ask called with empty 'text' field.")
            return JsonResponse({"error": "Empty text"}, status=400)

        # Validate and limit user input length
        if len(user_text) > 1000:
            logger.warning(f"User text too long: {len(user_text)} characters")
            return JsonResponse({"error": "Text too long. Maximum 1000 characters allowed."}, status=400)

        selected_domain = (payload.get("domain") or "normal").strip().lower()
        if selected_domain in ("healthcare", "finance", "normal"):
            request.session["selected_domain"] = selected_domain
        else:
            selected_domain = request.session.get("selected_domain", "normal")
        
        logger.info(f"Processing user query in domain: '{selected_domain}'. User text: '{user_text[:50]}...'")

        # Get and manage chat history with proper limits
        history = request.session.get("chat_history", [])
        
        # Limit history to prevent session bloat (keep last 20 messages = 10 exchanges)
        MAX_HISTORY_LENGTH = 20
        if len(history) > MAX_HISTORY_LENGTH:
            history = history[-MAX_HISTORY_LENGTH:]
            logger.info(f"Trimmed chat history to {MAX_HISTORY_LENGTH} messages")
        
        # Add user message to history
        history.append({"role": "user", "content": user_text})

        # --- CONTEXT-AWARE PROMPT ENGINEERING ---
        # Create a summary of the last few exchanges (only if history exists)
        recent_messages = history[-5:] if len(history) > 1 else history
        conversation_summary = "\n".join(
            [f"- {msg['role']}: {msg['content'][:100]}..." if len(msg['content']) > 100 else f"- {msg['role']}: {msg['content']}" 
             for msg in recent_messages]
        )

        base_personality = f"""You are Bodhita AI, an expert voice assistant. This is a continuous conversation.

CURRENT CONTEXT:
- This is a voice-based chat. Keep responses concise (2-3 sentences).
- Your response MUST be a direct continuation of the previous conversation.
- ALWAYS respond in HINGLISH (Roman script). NEVER use Devanagari.
- Recent conversation turns:
{conversation_summary}

YOUR TASK:
1. Acknowledge the user's latest message in the context of the history.
2. If the user is asking a follow-up question, connect your answer to what was discussed. Use phrases like "Jaise hum baat kar rahe the..." or "Uske baare mein aur batane ke liye...".
3. If the user changes the topic, acknowledge it and answer the new question.
4. Provide accurate, factual information in a friendly, conversational tone.
"""

        if selected_domain == "normal":
            system_prompt = base_personality
        else:
            kb_text = load_kb(selected_domain)
            if not kb_text:
                logger.warning(f"Knowledge base for domain '{selected_domain}' is empty")
                return JsonResponse({"error": f"Knowledge base for {selected_domain} is not available"}, status=503)
            
            system_prompt = f"""{base_personality}

KNOWLEDGE BASE INSTRUCTIONS:
- You are in {selected_domain.upper()} mode.
- You MUST answer ONLY using the Knowledge Base below.
- If the answer is not in the knowledge base, state that clearly, e.g., "Yeh jaankari mere knowledge base mein nahi hai."
- Refer to the conversation history for context, but derive your answer from the knowledge base.

--- {selected_domain.upper()} KNOWLEDGE BASE ---
{kb_text}
---
"""

        messages = [{"role": "system", "content": system_prompt}] + history

        # Store response outside generator for session management
        collected_response = {"text": "", "error": None}

        def generate_stream():
            """Generator function that streams response chunks to client."""
            client = get_azure_client()
            try:
                logger.info(f"Requesting chat completion with context. History length: {len(history)}.")
                stream = client.chat.completions.create(
                    model=MyConfig.envFile()["AZURE_OPENAI_DEPLOYMENT_NAME"],
                    messages=messages,
                    max_tokens=200,
                    temperature=0.7,
                    stream=True
                )

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        collected_response["text"] += content
                        yield f"data: {json.dumps({'chunk': content})}\n\n"

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                logger.info(f"Finished streaming response. Total length: {len(collected_response['text'])} chars")

            except Exception as e:
                logger.error(f"Error during response stream generation: {e}", exc_info=True)
                collected_response["error"] = str(e)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # Create the response with a generator wrapper that updates session after streaming
        def stream_with_session_update():
            """Wrapper generator that updates session after streaming completes."""
            yield from generate_stream()
            
            # Update session after all chunks have been sent
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
    """Clears the chat history and resets the conversation context."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    
    try:
        # Clear chat history and domain
        request.session["chat_history"] = []
        request.session.pop("selected_domain", None)
        request.session.modified = True
        
        logger.info("Chat history and context reset successfully")
        return JsonResponse({
            "status": "success", 
            "message": "Conversation history cleared"
        })
    except Exception as e:
        logger.error(f"Error resetting context: {e}", exc_info=True)
        return JsonResponse({"error": "Failed to reset context"}, status=500)

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
    """Handles text-to-speech conversion using Azure Speech Service."""
    if request.method != "POST":
        logger.warning(f"Received {request.method} request for api_tts, but only POST is allowed.")
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body)
        text = (payload.get("text") or "").strip()
        if not text:
            logger.warning("TTS API called with empty 'text' field.")
            return JsonResponse({"error": "Empty text"}, status=400)

        # Validate text length for TTS
        if len(text) > 5000:
            logger.warning(f"TTS text too long: {len(text)} characters")
            return JsonResponse({"error": "Text too long for synthesis. Maximum 5000 characters."}, status=400)

        logger.info(f"Requesting TTS for text: '{text[:50]}...'")
        synthesizer = get_speech_synthesizer()
        
        # Escape XML special characters to prevent SSML errors
        escaped_text = html.escape(text, quote=False)
        
        ssml = f"""<speak version='1.0' xml:lang='hi-IN'>
            <voice name='hi-IN-SwaraNeural'>
                <prosody rate='1.1' pitch='0%'>{escaped_text}</prosody>
            </voice>
        </speak>"""

        logger.debug(f"Generated SSML: {ssml[:200]}...")
        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            if not result.audio_data:
                logger.error("TTS returned empty audio data")
                return JsonResponse({"error": "No audio data generated"}, status=500)
            
            audio_base64 = base64.b64encode(result.audio_data).decode('utf-8')
            logger.info(f"Successfully synthesized audio. Size: {len(result.audio_data)} bytes")
            return JsonResponse({"audio": audio_base64, "format": "wav"})
        
        cancellation = result.cancellation_details
        error_msg = f"Reason: {cancellation.reason}, Details: {cancellation.error_details}"
        logger.error(f"TTS synthesis failed. {error_msg}")
        return JsonResponse({"error": "Speech synthesis failed", "details": error_msg}, status=500)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in TTS request body: {e}")
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Unexpected error in api_tts view: {e}", exc_info=True)
        return JsonResponse({"error": "Internal server error"}, status=500)
