from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

conversation_history = [
    {
        "role": "system",
        "content": (
            "You are MentorBot, a friendly senior student mentor. "
            "Explain everything clearly for a beginner. "
            "Use short bold headings using <strong>Heading</strong>. "
            "Use bullet points with <ul><li>. "
            "Do NOT use # or * symbols. "
            "Do NOT add extra spaces between words. "
            "Write natural English sentences like real chat."
        ),
    }
]

@app.get("/")
def home():
    return {"message": "MentorBot API is running!"}

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
                yield part  # ✅ NO CLEANING, NO SPACE CHANGES

        conversation_history.append(
            {"role": "assistant", "content": full_reply}
        )

    return StreamingResponse(generate(), media_type="text/plain")
