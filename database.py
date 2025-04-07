from sqlalchemy import create_engine, Column, Integer, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import text
import datetime
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL')

engine = create_engine(DATABASE_URL)

Base = declarative_base()

class Evaluation(Base):
    __tablename__ = 'evaluations'

    id = Column(Integer, primary_key=True)
    session_id = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    prompt = Column(Text)
    response = Column(Text)
    latency = Column(Float)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)

class ReferenceEvaluation(Base):
    __tablename__ = 'reference_evaluations'

    id = Column(Integer, primary_key=True)
    prompt = Column(Text)
    reference = Column(Text)
    model_response = Column(Text)
    rouge1 = Column(Float)
    rouge2 = Column(Float)
    rougel = Column(Float)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)

def save_evaluation(session_id, prompt, response, latency, usage):
    session = Session()
    new_eval = Evaluation(
        session_id=session_id,
        prompt=prompt,
        response=response,
        latency=latency,
        prompt_tokens=usage.get('prompt_tokens'),
        completion_tokens=usage.get('completion_tokens'),
        total_tokens=usage.get('total_tokens')
    )
    session.add(new_eval)
    session.commit()
    session.close()

def save_reference_evaluation(prompt, reference, model_response, rouge_scores):
    session = Session()
    new_ref_eval = ReferenceEvaluation(
        prompt=prompt,
        reference=reference,
        model_response=model_response,
        rouge1=rouge_scores.get('rouge1'),
        rouge2=rouge_scores.get('rouge2'),
        rougel=rouge_scores.get('rougeL')
    )
    session.add(new_ref_eval)
    session.commit()
    session.close()


def load_all_evaluations(session_id):
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM evaluations WHERE session_id = :sid'), {'sid': session_id})
        rows = result.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=result.keys())
            return df
        else:
            return pd.DataFrame()
        
def load_metric_evaluations():
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM reference_evaluations'))
        rows = result.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=result.keys())
            return df
        else:
            return pd.DataFrame()