import os
from dotenv import load_dotenv

load_dotenv()

# Paths — use /data/lancedb on Railway (persistent volume), local otherwise
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.getenv("DB_PATH", os.path.join(BASE_DIR, "data", "lancedb"))
TABLE_NAME = "fashion_kb"

# Embedding
EMBEDDING_MODEL = "clip-ViT-B-32"
EMBEDDING_DIM = 512

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
DEFAULT_TOP_K = 5

# Scraper schedule (minutes)
SCRAPE_INTERVAL_MINUTES = 60

# Ollama (local fallback)
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

# Groq
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"

# Conversation
MAX_HISTORY_TURNS = 5

# RSS feeds to scrape
RSS_FEEDS = [
    "https://www.vogue.com/feed/rss",
    "https://www.harpersbazaar.com/rss/all.xml/",
    "https://www.whowhatwear.com/rss",
    "https://news.google.com/rss/search?q=fashion+trends&hl=en-US&gl=US&ceid=US:en",
]

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
