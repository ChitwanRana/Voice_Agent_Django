from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from openai import AzureOpenAI
from src.config.config import MyConfig
from src.prompts.system_prompt import VOICE_ASSISTANT_PROMPT

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

        config = MyConfig.envFile()
        client = AzureOpenAI(
            api_key=config["AZURE_OPENAI_KEY"],
            api_version=config["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=config["AZURE_OPENAI_ENDPOINT"]
        )
        deployment_name = config["AZURE_OPENAI_DEPLOYMENT_NAME"]

        resp = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": VOICE_ASSISTANT_PROMPT},
                {"role": "user", "content": user_text}
            ],
            max_tokens=300
        )
        agent_reply = resp.choices[0].message.content
        return JsonResponse({"reply": agent_reply})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

