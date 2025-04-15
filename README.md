# Mistral 7B Instruct Summarization App
Check out the interactive app on [Streamlit](https://robibeywpi-mistral-llm-project-main-raz1lw.streamlit.app/)

This is an interactive text summarization or chat application that uses the [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) model via the Hugging Face Inference API. The app allows users to input text prompts and receive generated responses through the Streamlit front-end in seconds.

Currently this app uses free or low-cost versions of data storage and model access for academic use that are limited in amount of spend and usage to prevent misuse. In the public version, the user would need to generate their own API key for access to the model / pay for persistent storage.

## Features

- Generate summaries/responses from user inputs
- Track sessions with session IDs
- Store history in a PostgreSQL database
- Evaluated summarization performance on the XSum dataset using ROUGE and METEOR scores
- Clean and response UI built with Streamlit and hosted on Streamlit Cloud
- View usage metrics over time with a built in chart that updates automatically

## Technologies Used

- Python 3.13
- [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) - model
- [Hugging Face Hub](https://huggingface.co/docs/hub/en/index) - to query the model hosted on Hugging Face via the Inference Client
- [Streamlit](https://streamlit.io/) - frontend and deployment with [Streamlit Cloud](https://streamlit.io/cloud)
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/en/index) - compute ROUGE and METEOR scores
- [AWS RDS](https://aws.amazon.com/rds/) (PostgreSQL) - persistent storage
- SQLAlchemy - database logic in Python
- [XSum](https://paperswithcode.com/dataset/xsum) - reference dataset

## Project Structure

```
app-directory/
├─ main.py              # Streamlit frontend interface
├─ database.py          # SQLAlchemy model and DB functions
├─ reset_database.py    # DB reset and preloading reference prompts
├─ model.py             # Mistral 7B API querying
├─ requirements.txt     # Project dependencies
├─ .gitignore           # Ignore .env secrets file
├─ README.md            # Project overview
```