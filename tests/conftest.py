import os

# Set a dummy key before any test file is imported so that data_loader.py's
# module-level `client = OpenAI()` does not raise at collection time.
# A real key from .env (if present) takes precedence via load_dotenv(); either
# way, no real API calls are made in tests — those are all mocked.
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key-not-a-real-credential")
