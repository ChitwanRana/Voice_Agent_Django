import json
import logging
import httpx
from django.shortcuts import render
from django.http import JsonResponse
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


def index(request):
    return render(request, "voice_app/index.html")   # <— REQUIRED


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

        # Chat history
        history = request.session.get("chat_history", [])
        history.append({"role": "user", "content": user_text})

        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]

        # Prepare system prompt
        if selected_domain == "normal":
            system_prompt = VOICE_ASSISTANT_PROMPT
        else:
            kb_text = load_kb(selected_domain)
            system_prompt = (
                f"{VOICE_ASSISTANT_PROMPT}\n\n"
                f"You must answer ONLY using the {selected_domain} knowledge base below.\n"
                f"If answer missing, reply: "
                f"'I don’t have enough information in the {selected_domain} knowledge base to answer that.'\n\n"
                f"--- KB START ---\n{kb_text}\n--- KB END ---"
            )

        messages = [{"role": "system", "content": system_prompt}] + history

        # Azure Client
        config = MyConfig.envFile()
        client = AzureOpenAI(
            api_key=config["AZURE_OPENAI_KEY"],
            api_version=config["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=config["AZURE_OPENAI_ENDPOINT"],
            http_client=httpx.Client(timeout=15)
        )

        response = client.chat.completions.create(
            model=config["AZURE_OPENAI_DEPLOYMENT_NAME"],
            messages=messages,
            max_tokens=150,
            temperature=0.2
        )

        assistant_message = response.choices[0].message.content

        # Save to session
        history.append({"role": "assistant", "content": assistant_message})
        request.session["chat_history"] = history

        return JsonResponse({"reply": assistant_message})

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
    logger.info("Chat context successfully reset.")

    return JsonResponse({"status": "context reset"})
