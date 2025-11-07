from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

class MyConfig:
    """Configuration loader for Azure and other services."""

    @staticmethod
    def envFile():
        """Return environment variables as a dictionary."""
        return {
            # Azure Speech
            "SPEECH_KEY": os.getenv("SPEECH_KEY"),
            "SPEECH_REGION": os.getenv("SPEECH_REGION"),

            # Azure OpenAI
            "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY"),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),

            # Optional: Gemini
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        }
