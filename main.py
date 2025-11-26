from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os
import re

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


# ✅ CLEAN MARKDOWN FUNCTION (REMOVES #, *, BULLETS, ETC.)
def clean_markdown(text: str) -> str:
    text = re.sub(r"[*_#>`]", "", text)          # remove *, _, #, >, `
    text = re.sub(r"\n\s*-\s*", "\n", text)      # remove bullet dashes
    text = re.sub(r"\n{2,}", "\n\n", text)       # clean extra newlines
    return text.strip()


# ✅ SYSTEM PROMPT (PLAIN TEXT ONLY)
conversation_history = [
    {
        "role": "system",
        "content": (
            "You are MentorBot, a friendly senior-student mentor. "
            "Explain everything very clearly, simply, and step by step. "
            "Always assume the user is a complete beginner. "
            "IMPORTANT RULE: Reply in plain text only. "
            "Do NOT use markdown, headings, bullet points, #, *, or special formatting. "
            "Write like a real human chatting in WhatsApp style with simple sentences."
        ),
    }
]


@app.get("/")
def home():
    return {"message": "MentorBot API is running!"}


# ✅ NORMAL CHAT (NON-STREAM)
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

    raw_reply = response.choices[0].message.content
    clean_reply = clean_markdown(raw_reply)

    conversation_history.append(
        {"role": "assistant", "content": clean_reply}
    )

    return {"reply": clean_reply}


# ✅ STREAMING CHAT (CHATGPT STYLE TYPING, PLAIN TEXT ONLY)
@app.post("/chat-stream")
async def chat_stream(request: ChatRequest):

    # Add user message to memory
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
                cleaned_part = clean_markdown(part)

                full_reply += cleaned_part
                yield cleaned_part   # send cleaned text to UI

        # Save cleaned reply in memory
        conversation_history.append(
            {"role": "assistant", "content": full_reply}
        )

    return StreamingResponse(generate(), media_type="text/plain")
