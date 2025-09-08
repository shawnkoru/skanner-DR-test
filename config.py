import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "o4-mini-deep-research")
PARALLEL_AI_API_KEY = os.getenv("PARALLEL_AI_API_KEY")
