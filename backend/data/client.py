import os

from tiingo import TiingoClient
from dotenv import load_dotenv

load_dotenv()

TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

config = {}
config["session"] = True
config["api_key"] = TIINGO_API_KEY

tiingo_client = TiingoClient()
