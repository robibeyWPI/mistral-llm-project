from sqlalchemy import create_engine, text
from database import Base, EvaluationDB
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from model import query_mistral_chat
from datasets import load_dataset
import time
import evaluate

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL')
engine = create_engine(DATABASE_URL)

HF_TOKEN = os.environ.get('HF_TOKEN')
client = InferenceClient(api_key=HF_TOKEN)

db = EvaluationDB(DATABASE_URL)

meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

def truncate_text(text, max_words=200):
    words = text.split()
    return ' '.join(words[:max_words])

def preload_evaluation_data():
    print('Loading XSum dataset...')
    
    dataset = load_dataset('xsum', split='test', trust_remote_code=True)
    samples = dataset.select(range(100))
    
    
    for idx, sample in enumerate(samples, start=1):
        print(f'Generating ROUGE and METEOR scores for reference prompt {idx}')

        truncated_article = truncate_text(sample['document'])
        prompt = f' In one concise and short sentence, summarize the following article directly, starting with the main point. Do not begin with phrases like "This article" or "The article":\n\n{truncated_article}'
        reference_text = sample['summary']

        usage, response = query_mistral_chat(prompt)

        meteor_score = meteor.compute(predictions=[response], references=[sample['summary']])

        rouge_score = rouge.compute(predictions=[response], references=[sample['summary']])


        db.save_reference_evaluation(
            prompt=prompt,
            reference=reference_text,
            model_response=response,
            rouge_scores=rouge_score,
            meteor_score=meteor_score
        )
        time.sleep(0.5)

with engine.connect() as conn:
    conn.execute(text('DROP TABLE IF EXISTS evaluations'))
    conn.execute(text('DROP TABLE IF EXISTS reference_evaluations'))
    conn.commit()

print('Tables dropped.')

Base.metadata.create_all(engine)
preload_evaluation_data()
print('Tables recreated.')