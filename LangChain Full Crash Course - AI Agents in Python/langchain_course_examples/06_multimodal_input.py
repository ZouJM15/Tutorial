import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage


load_dotenv()

MODEL = os.getenv("VISION_MODEL", os.getenv("CHAT_MODEL", "openai:gpt-4.1-mini"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_IMAGE = Path(os.getenv("LOCAL_IMAGE", PROJECT_ROOT / "input.jpg"))
IMAGE_URL = os.getenv("IMAGE_URL")


def local_image_block(path: Path) -> dict:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image", "base64": encoded, "mime_type": mime_type}


def url_image_block(url: str) -> dict:
    return {"type": "image", "url": url}


model = init_chat_model(MODEL, temperature=0.1)
image_block = url_image_block(IMAGE_URL) if IMAGE_URL else local_image_block(LOCAL_IMAGE)

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the contents of this image."},
        image_block,
    ]
)

response = model.invoke([message])

print(response.content)
