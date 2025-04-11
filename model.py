from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

# Decided to switch from Replicate API to HuggingFace. Replicate API did not provide the option to deploy my own Mistral-7B model, I could not access it. HuggingFace turned out to be the better option anyway. Free (small limit) and much faster than I was expecting
HF_TOKEN = os.environ.get('HF_TOKEN')
client = InferenceClient(
    provider='hf-inference',
    api_key=HF_TOKEN
)


def query_mistral_chat(prompt):
    completion = client.chat.completions.create(
        model='mistralai/Mistral-7B-Instruct-v0.3', # The instruct model is better at text summarization, which is the goal here
        messages=[
            {'role': 'user',
             'content': prompt,
            }
        ],
        max_tokens=500, # 500 is the number used in the docs
        stream_options={'include_usage': True} # Returns token usage
    )
    token_usage: dict = completion.usage
    content: str = completion.choices[0].message.content
    return token_usage, content