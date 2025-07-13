import os
from dotenv import load_dotenv
from alpaca_trade_api.stream import Stream

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")

stream = Stream(
    API_KEY,
    API_SECRET,
    base_url=BASE_URL,
    data_feed='iex'  
)

@stream.on_trade_updates
async def handle_trade_update(data):
    print("🔔 Trade update:", data)

if __name__ == "__main__":
    stream.run()
