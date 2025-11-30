import json
import logging
import httpx
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from openai import AzureOpenAI
from django.core.cache import cache
from src.config.config import MyConfig
from src.prompts.system_prompt import VOICE_ASSISTANT_PROMPT
from pathlib import Path

logger = logging.getLogger("voice_app")

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
        config = MyConfig.envFile()
        _azure_client = AzureOpenAI(
            api_key=config["AZURE_OPENAI_KEY"],
            api_version=config["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=config["AZURE_OPENAI_ENDPOINT"],
            http_client=httpx.Client(
                timeout=10,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        )
    return _azure_client


def index(request):
    return render(request, "voice_app/index.html")


def load_kb(domain):
    kb = cache.get(domain)
    if kb:
        return kb
    fp = KB_FILES.get(domain)
    content = fp.read_text("utf8")
    cache.set(domain, content, timeout=None)
    return content


@csrf_exempt
def api_ask(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        payload = json.loads(request.body)
        user_text = (payload.get("text") or "").strip()
        selected_domain = (payload.get("domain") or "").strip().lower()

        if selected_domain in ("healthcare", "finance", "normal"):
            request.session["selected_domain"] = selected_domain
        else:
            selected_domain = request.session.get("selected_domain", "normal")

        if not user_text:
            return JsonResponse({"error": "Empty text"}, status=400)

        # Chat history - limit to last 6 messages
        history = request.session.get("chat_history", [])
        history.append({"role": "user", "content": user_text})

        if len(history) > 6:
            history = history[-6:]

        # Prepare system prompt
        if selected_domain == "normal":
            system_prompt = VOICE_ASSISTANT_PROMPT
        else:
            kb_text = load_kb(selected_domain)
            system_prompt = (
                f"{VOICE_ASSISTANT_PROMPT}\n\n"
                f"You must answer ONLY using the {selected_domain} knowledge base below.\n"
                f"If answer missing, reply: "
                f"'I don't have enough information in the {selected_domain} knowledge base to answer that.'\n\n"
                f"--- KB START ---\n{kb_text}\n--- KB END ---"
            )

        messages = [{"role": "system", "content": system_prompt}] + history

        # Stream generator
        def generate_stream():
            client = get_azure_client()
            full_response = ""
            
            try:
                stream = client.chat.completions.create(
                    model=MyConfig.envFile()["AZURE_OPENAI_DEPLOYMENT_NAME"],
                    messages=messages,
                    max_tokens=100,
                    temperature=0.1,
                    stream=True
                )

                for chunk in stream:
                    # Safety check: ensure choices exists and has content
                    if (hasattr(chunk, 'choices') and 
                        len(chunk.choices) > 0 and 
                        hasattr(chunk.choices[0], 'delta') and 
                        hasattr(chunk.choices[0].delta, 'content') and
                        chunk.choices[0].delta.content):
                        
                        content = chunk.choices[0].delta.content
                        full_response += content
                        # Send SSE format
                        yield f"data: {json.dumps({'chunk': content})}\n\n"

                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"

                # Save to session
                if full_response:
                    history.append({"role": "assistant", "content": full_response})
                    request.session["chat_history"] = history
                    request.session.modified = True

            except Exception as e:
                logger.exception("Error in stream generation")
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
        logger.exception(e)
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
