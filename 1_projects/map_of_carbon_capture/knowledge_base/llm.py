# Using LLM to generate topics
# Improvement: add rate limiting error handling so you don't hardcode the wait time

# imports
import dotenv
import os
import openai

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_GPT4_API_KEY")
# openai.api_key = os.getenv("OPENAI_API_KEY")

# models
EMBEDDING_MODEL = "text-embedding-ada-003"
# GPT_MODEL = "gpt-3.5-turbo"
GPT_MODEL = "gpt-4"

# for bulk openai message, no stream
def chat_openai(prompt="Tell me to ask you a prompt", model=GPT_MODEL, chat_history=[]):
    # define message conversation for model
    if chat_history:
        messages = chat_history
    else:
        messages = [
            {"role": "system", "content": "You are an educated carbon capture research consultant and a generally educated and helpful researcher and programmer. Answer as correctly, clearly, and concisely as possible."},
        ]
    messages.append({"role": "user", "content": prompt})

    # create the chat completion
    print("Prompt: ", prompt)
    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    print("Completion info: ", completion)
    text_answer = completion["choices"][0]["message"]["content"]

    # updated conversation history
    messages.append({"role": "assistant", "content": text_answer})

    return text_answer, messages