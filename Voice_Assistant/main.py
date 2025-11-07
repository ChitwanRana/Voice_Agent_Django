import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from src.config.config import MyConfig
from src.prompts.system_prompt import VOICE_ASSISTANT_PROMPT
import logging
import os
from datetime import datetime


logs_dir = os.path.join("src","logs")
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Configure logging
log_filename = os.path.join(logs_dir, f"voice_agent.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load config
config = MyConfig.envFile()

logger.info("Checking Azure Speech Service configuration...")
logger.info(f"Region: {config.get('SPEECH_REGION', 'NOT_SET')}")
logger.info(f"Key present: {'Yes' if config.get('SPEECH_KEY') else 'No'}")
logger.info(f"Key length: {len(config.get('SPEECH_KEY', '')) if config.get('SPEECH_KEY') else 0}")

speech_config = speechsdk.SpeechConfig(
    subscription=config["SPEECH_KEY"],
    region=config["SPEECH_REGION"]
)

# language of the bot, english by default
speech_config.speech_recognition_language = "en-US"

# Using default voice (no specific voice name set)
logger.info("Voice set to: Default system voice")

# Set speech synthesis output to default speaker
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

client = AzureOpenAI(
    api_key=config["AZURE_OPENAI_KEY"],
    api_version=config["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=config["AZURE_OPENAI_ENDPOINT"]
)
deployment_name = config["AZURE_OPENAI_DEPLOYMENT_NAME"]
logger.info(f"Azure OpenAI deployment: {deployment_name}")

# Speech recognizer + synthesizer
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

print("Speak to the AI agent (say 'exit' to stop)...")
print("Make sure your microphone is connected and working...")
logger.info("Voice agent started")



while True:
    try:
        print("Listening...")
        logger.info("Listening for user input...")
        result = speech_recognizer.recognize_once_async().get()
        
        # Check recognition result status
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            user_text = result.text.strip()
            print(f"You said: {user_text}")
            logger.info(f"User input recognized: {user_text}")
            
            if not user_text:
                print("No speech detected. Please speak clearly.")
                logger.warning("Empty speech detected")
                continue
                
            if "exit" in user_text.lower():
                print("Exiting...")
                logger.info("Exit command received")
                break
                
            # Get AI response
            logger.info("Requesting AI response...")
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": VOICE_ASSISTANT_PROMPT},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=150
            )
            
            agent_reply = response.choices[0].message.content
            print(f"AI: {agent_reply}")
            logger.info(f"AI response: {agent_reply}")
            
            # Wait for speech synthesis to complete
            print("AI is speaking...")
            logger.info("Starting speech synthesis...")
            speech_result = speech_synthesizer.speak_text_async(agent_reply).get()
            
            # Check if speech synthesis was successful
            if speech_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("AI finished speaking")
                logger.info("Speech synthesis completed successfully")
            elif speech_result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speech_result.cancellation_details
                logger.error(f"Speech synthesis canceled: {cancellation_details.reason}")
                logger.error(f"Error details: {cancellation_details.error_details}")
                print(f"Speech synthesis canceled: {cancellation_details.reason}")
                print(f"Error: {cancellation_details.error_details}")
            
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized. Please try again.")
            logger.warning("No speech recognized")
            
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.error(f"Speech recognition canceled: {cancellation_details.reason}")
            print(f"Speech recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Error details: {cancellation_details.error_details}")
                print(f"Error details: {cancellation_details.error_details}")
                break
                
    except KeyboardInterrupt:
        print("\nExiting...")
        logger.info("Keyboard interrupt received - exiting gracefully")
        break
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"An error occurred: {e}")
        break

logger.info("Voice agent stopped")
print("Voice agent stopped")
