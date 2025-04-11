import streamlit as st
import time
from model import query_mistral_chat
from database import EvaluationDB
import uuid
import altair # custom chart
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL')

query_params = st.query_params # Params from the URL
session_id = query_params.get('session_id', [None])[0] # Look for a session ID within the query params dictionary

if not session_id:
    session_id = str(uuid.uuid4()) # If it doesn't exist, we make one and set it in the next 2 lines
    st.query_params['session_id'] = session_id
st.session_state.session_id = session_id

db = EvaluationDB(DATABASE_URL)

st.title('Mistral 7B App')
tab1, tab2, tab3 = st.tabs(['Chat with Mistral 7B Instruct', 'History', 'Evaluation Metrics'])

# Chat area
with tab1:
    user_input = st.text_area('Enter your prompt:')
    if st.button('Submit'):
        with st.spinner('Generating response...'):
            start_time = time.perf_counter()
            usage, response = query_mistral_chat(user_input)
            end_time = time.perf_counter()
            inf_time = end_time - start_time
            db.save_evaluation(
                session_id=st.session_state.session_id,
                prompt=user_input,
                response=response,
                latency=inf_time,
                usage=usage
            )

            st.markdown('**Response:**')
            st.write(response)
            with st.expander('See metrics for this prompt and response:'):
                st.markdown(f'**Inference time:**')
                st.write(f'{inf_time:.2f} seconds')
                st.markdown(f'**Token usage:**')
                for key, val in usage.items():
                    st.write(f'{key}: {val}')

# User history
with tab2:
    df = db.load_all_evaluations(st.session_state.session_id)

    if not df.empty:
        if st.button('Clear history'):
            db.clear_all_chats(st.session_state.session_id)
            st.rerun()
        df_copy = df.drop(['id', 'session_id'], axis=1)
        st.dataframe(df_copy)

    else: st.info('Submit your first prompt to show your history.')

# ROUGE metrics and token count for users
with tab3:
    static_df = db.load_metric_evaluations()

    if not static_df.empty:
        with st.container(border=True):
            st.subheader('ROUGE score on reference text')
            col1, col2, col3 = st.columns(3)
            col1.metric('**Rouge1 Avg**', f'{static_df['rouge1'].mean():.2f}', border=True)
            col2.metric('**Rouge2 Avg**', f'{static_df['rouge2'].mean():.2f}', border=True)
            col3.metric('**RougeL Avg**', f'{static_df['rougel'].mean():.2f}', border=True)

            st.subheader('METEOR score on reference text')
            st.metric('**Meteor Avg**', f'{static_df['meteor'].mean():.2f}', border=True)
            st.dataframe(static_df.drop(['id'], axis=1))


        if not df.empty:
            with st.container(border=True):
                st.subheader('Metrics on your history')
                col1, col2, col3, col4 = st.columns(4)
                col1.metric('**Latency Avg**', f'{df['latency'].mean():.2f} sec', border=True)
                col2.metric('**Prompt Tokens Avg**', f'{df['prompt_tokens'].mean():.2f}', border=True)
                col3.metric('**Response Tokens Avg**', f'{df['completion_tokens'].mean():.2f}', border=True)
                col4.metric('**Total Tokens Avg**', f'{df['total_tokens'].mean():.2f}', border=True)

                df['count'] = range(1, len(df) + 1)
                base = altair.Chart(df).mark_line().encode(
                x=altair.X('count:O', title='Prompts Count', axis=altair.Axis(labelAngle=0)),
                y=altair.Y('sum(total_tokens):Q', title='Total Tokens')
            ).properties(
                title=altair.Title(
                    text='Cumulative Tokens Used',
                    anchor='middle'
                )
            )
                line = base.mark_line()
                points = base.mark_point(filled=True, size=70, shape='circle')
                chart = line + points
                st.altair_chart(chart, use_container_width=True)
        else: st.info('Come back after you submit your first prompt to see personal usage statistics.')

    else: st.warning('Evaluation metrics have not been calculated yet. Please check back later.')