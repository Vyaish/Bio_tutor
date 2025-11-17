from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load env vars
load_dotenv(override=True)

# Connect to Ollama
ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# --- SYSTEM PROMPT ---
system_prompt = """
You are a helpful biology tutor who has a phD in biology
Answer clearly, accurately, and with examples where helpful.
If a user asks a high-school-level biology question, keep it simple.
If a user asks an advanced biology question, provide detailed, scientific explanations.
"""

# --- CHAT FUNCTION ---
def chat(message, history):
    # Build messages list for Ollama
    messages = [{"role": "system", "content": system_prompt}]
    
    for past_user, past_bot in history:
        messages.append({"role": "user", "content": past_user})
        messages.append({"role": "assistant", "content": past_bot})
    
    messages.append({"role": "user", "content": message})

    # Get response from Ollama
    response = ollama.chat.completions.create(
        model="llama3.2",
        messages=messages
    )

    return response.choices[0].message.content

# --- GRADIO CHAT UI ---
gr.ChatInterface(chat).launch(share=True)
