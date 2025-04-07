import streamlit as st
import time
from model import query_mistral_chat
from database import save_evaluation, load_all_evaluations, load_metric_evaluations
import uuid

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
tab1, tab2, tab3 = st.tabs(['Chat with Mistral 7B Instruct', 'History', 'Evaluation Metrics'])

with tab1:
    user_input = st.text_area('Enter your prompt:')
    if st.button('Submit'):
        with st.spinner('Generating response...'):
            start_time = time.perf_counter()
            usage, response = query_mistral_chat(user_input)
            end_time = time.perf_counter()
            inf_time = end_time - start_time
            save_evaluation(
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

with tab2:
    df = load_all_evaluations(st.session_state.session_id)

    if not df.empty:
        df = df.drop(['id', 'session_id'], axis=1)
        st.dataframe(df)

    else: st.warning('Empty DF')

with tab3:
    df = load_metric_evaluations()

    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric('**Rouge1 Avg**', f'{df['rouge1'].mean():.2f}', border=True)
        col2.metric('**Rouge2 Avg**', f'{df['rouge2'].mean():.2f}', border=True)
        col3.metric('**RougeL Avg**', f'{df['rougel'].mean():.2f}', border=True)
        st.dataframe(df)

    else: st.warning('Evaluation metrics have not been calculated yet.')