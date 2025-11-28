import json
import logging
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openai import AzureOpenAI
from src.config.config import MyConfig
from src.prompts.system_prompt import VOICE_ASSISTANT_PROMPT

logger = logging.getLogger("voice_app")  # custom logger

MAX_HISTORY = 40  # trim session messages


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

        messages = [{"role": "system", "content": VOICE_ASSISTANT_PROMPT}] + chat_history

        # Call Azure OpenAI
        resp = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=300,
            temperature=0.7
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
