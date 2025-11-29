import json
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openai import AzureOpenAI
from src.config.config import MyConfig
from src.prompts.system_prompt import VOICE_ASSISTANT_PROMPT

from pathlib import Path  # NEW

logger = logging.getLogger("voice_app")  # custom logger

MAX_HISTORY = 40  # trim session messages

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"  # NEW
KB_FILES = {
    "healthcare": KB_DIR / "healthcare.md",
    "finance": KB_DIR / "finance.md",
}


def load_kb(domain: str) -> str:
    """Load KB text for a domain; return empty string if missing."""
    fp = KB_FILES.get(domain)
    if not fp or not fp.exists():
        logger.warning("KB file missing for domain: %s", domain)
        return ""
    try:
        return fp.read_text(encoding="utf-8")
    except Exception as e:
        logger.exception("Failed reading KB for %s: %s", domain, str(e))
        return ""


def index(request):
    logger.info("Rendering voice assistant index page")
    return render(request, "voice_app/index.html")


@csrf_exempt
def api_ask(request):
    logger.info("api_ask endpoint hit. Method: %s", request.method)

    if request.method != "POST":
        logger.warning("Invalid method used on api_ask: %s", request.method)
        return JsonResponse({"error": "POST required."}, status=405)

    try:
        raw_data = request.body.decode("utf-8")
        logger.debug("Raw request body: %s", raw_data)

        payload = json.loads(raw_data)
        user_text = (payload.get("text") or "").strip()
        selected_domain = (payload.get("domain") or "").strip().lower()  # NEW

        if selected_domain in ("healthcare", "finance", "normal"):  # UPDATED
            request.session["selected_domain"] = selected_domain
            logger.info("Domain set from request: %s", selected_domain)
        else:
            selected_domain = request.session.get("selected_domain", "normal")  # UPDATED default normal
            logger.info("Domain fallback used: %s", selected_domain)

        if not user_text:
            logger.warning("Received empty user text")
            return JsonResponse({"error": "Empty text."}, status=400)

        logger.info("User text received: %s", user_text)

        # Retrieve chat history
        chat_history = request.session.get("chat_history", [])
        logger.debug("Current chat history length: %d", len(chat_history))

        chat_history.append({"role": "user", "content": user_text})
        logger.debug("User message appended. New history length: %d", len(chat_history))

        # Trim old messages
        if len(chat_history) > MAX_HISTORY:
            logger.info("Chat history exceeded max size. Trimming to last %d messages.", MAX_HISTORY)
            chat_history = chat_history[-MAX_HISTORY:]

        # Load config
        config = MyConfig.envFile()
        logger.info("Loaded Azure config.")

        client = AzureOpenAI(
            api_key=config["AZURE_OPENAI_KEY"],
            api_version=config["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=config["AZURE_OPENAI_ENDPOINT"]
        )
        deployment_name = config["AZURE_OPENAI_DEPLOYMENT_NAME"]

        logger.info("Sending request to Azure OpenAI model: %s", deployment_name)

        # NEW: Build system prompt depending on domain
        if selected_domain == "normal":
            # No KB restriction; use original behavior
            system_prompt = VOICE_ASSISTANT_PROMPT
            logger.info("Using NORMAL domain (no KB restriction).")
        else:
            kb_text = load_kb(selected_domain)
            domain_instructions = (
                f"You are answering ONLY using the {selected_domain} knowledge base provided below. "
                f"If the answer is not contained in the knowledge base, reply: "
                f'\"I don’t have enough information in the {selected_domain} knowledge base to answer that.\". '
                f"Do not fabricate. Cite sections or headings when possible.\n\n"
                f"--- {selected_domain.upper()} KNOWLEDGE BASE ---\n{kb_text}\n--- END KB ---\n"
            )
            system_prompt = f"{VOICE_ASSISTANT_PROMPT}\n\n{domain_instructions}"
            logger.info("Using domain-scoped KB: %s", selected_domain)

        messages = [{"role": "system", "content": system_prompt}] + chat_history

        # Call Azure OpenAI
        resp = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=300,
            temperature=0.2  # lower temp to stick to KB
        )

        logger.info("Azure OpenAI response received successfully.")

        agent_reply = resp.choices[0].message.content
        logger.debug("Assistant reply: %s", agent_reply)

        chat_history.append({"role": "assistant", "content": agent_reply})
        request.session["chat_history"] = chat_history
        logger.info("Chat history updated in session.")

        return JsonResponse({"reply": agent_reply})

    except Exception as e:
        logger.exception("Error in api_ask: %s", str(e))  # logs complete traceback
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
