from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request model
class ChatRequest(BaseModel):
    message: str

# Conversation memory
conversation_history = [
    {
        "role": "system",
        "content": (
            "You are MentorBot, a friendly senior student mentor. "
            "Explain deeply but in a very simple beginner-friendly way. "
            "Use bold headings using <strong> tags. "
            "Use bullet points using <ul> and <li> HTML tags. "
            "Never use markdown symbols like # or *. "
            "Always add proper spacing between words and lines. "
            "Your response must be clean HTML that can be safely shown inside a chat bubble."
        ),
    }
]

# Health check
@app.get("/")
def home():
    return {"message": "MentorBot API is running!"}

# Normal chat
@app.post("/chat")
def chat(msg: dict):
    user_message = msg.get("message", "")

    conversation_history.append(
        {"role": "user", "content": user_message}
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=conversation_history
    )

    ai_reply = response.choices[0].message.content

    conversation_history.append(
        {"role": "assistant", "content": ai_reply}
    )

    return {"reply": ai_reply}

# Streaming chat (ChatGPT style)
@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):

    conversation_history.append(
        {"role": "user", "content": request.message}
    )

    def generate():
        stream = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=conversation_history,
            stream=True
        )

        full_reply = ""

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                part = chunk.choices[0].delta.content
                full_reply += part
                yield part  # ✅ DO NOT TOUCH SPACES

        conversation_history.append(
            {"role": "assistant", "content": full_reply}
        )

    return StreamingResponse(generate(), media_type="text/plain")
