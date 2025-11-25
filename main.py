from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load .env file
load_dotenv()

# FastAPI app
app = FastAPI()

# Allow frontend (index.html)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI with secure API key
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
            "Explain concepts very clearly, simply, and step-by-step. "
            "Never use complicated terms unless necessary, and always stay positive and motivating."
        ),
    }
]


@app.get("/")
def home():
    return {"message": "MentorBot API is running!"}


# Normal chat (non-stream)
@app.post("/chat")
def chat(msg: dict):
    user_message = msg.get("message", "")

    conversation_history.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=conversation_history
    )

    ai_reply = response.choices[0].message.content

    conversation_history.append({"role": "assistant", "content": ai_reply})

    return {"reply": ai_reply}


# Streaming chat (ChatGPT-style typing)
@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):

    # Add user message to memory
    conversation_history.append({"role": "user", "content": request.message})

    def generate():
        stream = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=conversation_history,
            stream=True
        )

        full_reply = ""

        for chunk in stream:
            if chunk.choices[0].delta.content:
                part = chunk.choices[0].delta.content
                full_reply += part
                yield part  # send to UI

        # Save the bot's full reply in memory
        conversation_history.append({"role": "assistant", "content": full_reply})

    return StreamingResponse(generate(), media_type="text/plain")
