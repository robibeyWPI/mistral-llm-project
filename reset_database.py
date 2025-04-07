from sqlalchemy import create_engine, text
from database import Base, save_reference_evaluation
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from model import query_mistral_chat, calculate_rouge_score

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL')
engine = create_engine(DATABASE_URL)

HF_TOKEN = os.environ.get('HF_TOKEN')
client = InferenceClient(api_key=HF_TOKEN)

reference_dict = {
    'Summarize the story of Little Red Riding Hood in a few sentences.': "Little Red Riding Hood is a classic fairy tale about a young girl who is sent by her mother to deliver food to her sick grandmother. Along the way, she encounters a cunning wolf who tricks her into revealing where her grandmother lives. The wolf reaches the grandmother’s house first, disguises himself, and deceives Little Red Riding Hood. Depending on the version of the story, a woodsman or hunter saves them, or Little Red Riding Hood must use her wits to escape. The tale teaches lessons about trusting strangers and the importance of caution.",

    'Summarize the plot of Romeo and Juliet in a few sentences.': "Romeo and Juliet is a tragic play by William Shakespeare about two young lovers from feuding families in Verona. Despite their families hatred, Romeo Montague and Juliet Capulet fall in love and secretly marry. Their efforts to unite their families fail, leading to misunderstandings and a series of tragic events. Romeo mistakenly believes Juliet is dead and takes his own life; upon waking, Juliet also dies by suicide. Their deaths ultimately reconcile their warring families.",
    
    'Summarize the importance of photosynthesis in a few sentences.': "Photosynthesis is the process by which green plants, algae, and some bacteria convert sunlight into chemical energy. During photosynthesis, organisms absorb carbon dioxide from the atmosphere and water from the soil, using sunlight captured by chlorophyll to produce glucose and oxygen. This process not only fuels plant growth but also provides oxygen essential for most life forms on Earth. Photosynthesis plays a crucial role in maintaining the planet’s atmosphere and supporting ecosystems.",
}

def preload_evaluation_data():
    print('Generating ROUGE scores for reference prompts.')

    for prompt, reference_text in reference_dict.items():
        usage, response = query_mistral_chat(prompt)
        rouge_scores = calculate_rouge_score(reference_text, response)

        save_reference_evaluation(
            prompt=prompt,
            reference=reference_text,
            model_response=response,
            rouge_scores=rouge_scores
        )

with engine.connect() as conn:
    conn.execute(text('DROP TABLE IF EXISTS evaluations'))
    conn.execute(text('DROP TABLE IF EXISTS reference_evaluations'))
    conn.commit()

print('Tables dropped.')

Base.metadata.create_all(engine)
preload_evaluation_data()
print('Tables recreated.')