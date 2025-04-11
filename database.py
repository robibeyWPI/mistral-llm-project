from sqlalchemy import create_engine, Column, Integer, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import text
import datetime
import pandas as pd

Base = declarative_base() # This way we don't have to create an empty Base(DeclarativeBase) class to use in the classes below

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
    meteor = Column(Float)

class EvaluationDB():
    def __init__(self, database_url):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine) # Creates the tables within the DB

    def save_evaluation(self, session_id, prompt, response, latency, usage):
        '''Creates a session in the DB and saves it with the proper format.'''

        session = self.Session()
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

    def save_reference_evaluation(self, prompt, reference, model_response, rouge_scores, meteor_score):
        '''Creates a session in the DB and adds it with the proper format. This is for the predetermined reference text with the ROUGE scores.'''

        session = self.Session()
        new_ref_eval = ReferenceEvaluation(
            prompt=prompt,
            reference=reference,
            model_response=model_response,
            rouge1=float(rouge_scores.get('rouge1')),
            rouge2=float(rouge_scores.get('rouge2')),
            rougel=float(rouge_scores.get('rougeL')),
            meteor=float(meteor_score.get('meteor'))
        )
        session.add(new_ref_eval)
        session.commit()
        session.close()


    def load_all_evaluations(self, session_id):
        '''Load each users' session and return a pandas dataframe of that session. The columns are the columns in the DB and the rows are the data values.'''

        with self.engine.connect() as conn:
            result = conn.execute(text('SELECT * FROM evaluations WHERE session_id = :sid'), {'sid': session_id}) # make sure each user only gets their own history
            rows = result.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=result.keys())
                return df
            else:
                return pd.DataFrame()
        
    def load_metric_evaluations(self):
        '''I decided to calculate ROUGE scores from predetermined reference text and insert them upon DB creation.
        This function inserts it into each users' metrics tab.'''

        with self.engine.connect() as conn:
            result = conn.execute(text('SELECT * FROM reference_evaluations'))
            rows = result.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=result.keys())
                return df
            else:
                return pd.DataFrame()
            
    def clear_all_chats(self, session_id):
        '''Allow the users to clear their history. Also makes more room in the DB.'''

        with self.engine.connect() as conn:
            conn.execute(text('DELETE FROM evaluations WHERE session_id = :sid'), {'sid': session_id})
            conn.commit()