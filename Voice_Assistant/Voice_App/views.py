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
    kb = cache.get(domain)
    if kb:
        logger.info(f"Knowledge base for domain '{domain}' found in cache.")
        return kb
    
    fp = KB_FILES.get(domain)
    if not fp or not fp.exists():
        logger.warning(f"Knowledge base file for domain '{domain}' not found or path is invalid.")
        return ""
    
    try:
        logger.info(f"Loading knowledge base for domain '{domain}' from file: {fp}")
        content = fp.read_text("utf8")
        cache.set(domain, content, timeout=None) # Cache indefinitely
        logger.info(f"Knowledge base for domain '{domain}' cached successfully.")
        return content
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

        selected_domain = (payload.get("domain") or "normal").strip().lower()
        if selected_domain in ("healthcare", "finance", "normal"):
            request.session["selected_domain"] = selected_domain
        else:
            selected_domain = request.session.get("selected_domain", "normal")
        
        logger.info(f"Processing user query in domain: '{selected_domain}'. User text: '{user_text}'")

        history = request.session.get("chat_history", [])
        history.append({"role": "user", "content": user_text})
        
        if len(history) > 10:
            history = history[-10:]

        # --- CONTEXT-AWARE PROMPT ENGINEERING ---
        # Create a summary of the last few exchanges to prime the model
        conversation_summary = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in history[-4:]])

        base_personality = f"""You are Bodhita AI, an expert voice assistant. This is a continuous conversation.

CURRENT CONTEXT:
- This is a voice-based chat. Keep responses concise (2-3 sentences).
- Your response MUST be a direct continuation of the previous conversation.
- ALWAYS respond in HINGLISH (Roman script). NEVER use Devanagari.
- Recent conversation turns:
{conversation_summary}

YOUR TASK:
1.  Acknowledge the user's latest message in the context of the history.
2.  If the user is asking a follow-up question, connect your answer to what was discussed. Use phrases like "Jaise hum baat kar rahe the..." or "Uske baare mein aur batane ke liye...".
3.  If the user changes the topic, acknowledge it and answer the new question.
4.  Provide accurate, factual information in a friendly, conversational tone.
"""

        if selected_domain == "normal":
            system_prompt = base_personality
        else:
            kb_text = load_kb(selected_domain)
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

        def generate_stream():
            client = get_azure_client()
            full_response = ""
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
                    if (chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content):
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield f"data: {json.dumps({'chunk': content})}\n\n"

                yield f"data: {json.dumps({'done': True, 'full_response': full_response})}\n\n"
                logger.info("Finished streaming response from Azure OpenAI.")

            except Exception as e:
                logger.error(f"Error during response stream generation: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # We handle the response outside the generator to properly manage session state
        response_stream = generate_stream()
        
        # Consume the generator to get the full response and update history
        full_response_from_stream = ""
        for data_chunk in response_stream:
            # This loop is now just for consumption, the client gets the data directly.
            # We need to find the full response from the 'done' message.
            if data_chunk.startswith("data:"):
                try:
                    data = json.loads(data_chunk[5:])
                    if data.get('done') and 'full_response' in data:
                        full_response_from_stream = data['full_response']
                        break
                except json.JSONDecodeError:
                    continue
        
        if full_response_from_stream:
            history.append({"role": "assistant", "content": full_response_from_stream})
            request.session["chat_history"] = history
            request.session.modified = True
            logger.info(f"Updated chat history. New length: {len(history)} messages.")

        # Return the original generator, which is now ready to be sent to the client
        return StreamingHttpResponse(
            generate_stream(),
            content_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
        )

    except Exception as e:
        logger.error(f"Unexpected error in api_ask view: {e}", exc_info=True)
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
            audio_base64 = base64.b64encode(result.audio_data).decode('utf-8')
            logger.info(f"Successfully synthesized audio. Size: {len(result.audio_data)} bytes")
            return JsonResponse({"audio": audio_base64, "format": "wav"})
        
        cancellation = result.cancellation_details
        logger.error(f"TTS synthesis failed. Reason: {cancellation.reason}. Details: {cancellation.error_details}")
        return JsonResponse({"error": f"Synthesis failed: {cancellation.reason}"}, status=500)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in TTS request body: {e}")
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Unexpected error in api_tts view: {e}", exc_info=True)
        return JsonResponse({"error": f"TTS error: {str(e)}"}, status=500)
