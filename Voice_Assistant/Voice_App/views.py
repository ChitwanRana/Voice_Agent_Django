from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from openai import AzureOpenAI
from src.config.config import MyConfig
from src.prompts.system_prompt import VOICE_ASSISTANT_PROMPT

MAX_HISTORY = 40  # trim session messages


def index(request):
    return render(request, "voice_app/index.html")


@csrf_exempt
def api_ask(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required."}, status=405)
    try:
        payload = json.loads(request.body.decode("utf-8"))
        user_text = (payload.get("text") or "").strip()
        if not user_text:
            return JsonResponse({"error": "Empty text."}, status=400)

        chat_history = request.session.get("chat_history", [])

        chat_history.append({"role": "user", "content": user_text})

        # Trim old messages (keep most recent)
        if len(chat_history) > MAX_HISTORY:
            chat_history = chat_history[-MAX_HISTORY:]

        config = MyConfig.envFile()
        client = AzureOpenAI(
            api_key=config["AZURE_OPENAI_KEY"],
            api_version=config["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=config["AZURE_OPENAI_ENDPOINT"]
        )
        deployment_name = config["AZURE_OPENAI_DEPLOYMENT_NAME"]

        messages = [{"role": "system", "content": VOICE_ASSISTANT_PROMPT}] + chat_history

        resp = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        agent_reply = resp.choices[0].message.content

        chat_history.append({"role": "assistant", "content": agent_reply})
        request.session["chat_history"] = chat_history

        return JsonResponse({"reply": agent_reply})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def reset_context(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required."}, status=405)
    request.session["chat_history"] = []
    return JsonResponse({"status": "context reset"})