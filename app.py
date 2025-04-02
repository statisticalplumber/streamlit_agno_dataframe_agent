import streamlit as st
import pandas as pd
from pathlib import Path
import time

from agno.agent import Agent
from agno.models.openai import OpenAILike
try:
    from dataframe_tools import DataFrameTools
except ImportError:
    st.error("Missing DataFrameTools.")
    st.stop()
from agno.utils.log import log_debug, log_info, logger

# Settings
MODEL_BASE_URL = "http://localhost:1234/v1"
MODEL_API_KEY = "lm-studio"

st.set_page_config(layout="wide", page_title="DataFrame Chat Agent")
st.title("📊 Agent LLM")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "dataframe_name" not in st.session_state:
    st.session_state.dataframe_name = "df"

# Sidebar - upload + setup
with st.sidebar:
    st.header("1. Load Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file and st.session_state.dataframe is None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.dataframe = df
            name = Path(uploaded_file.name).stem.replace("-", "_").replace(" ", "_")
            st.session_state.dataframe_name = name or "df"
            st.success(f"Loaded! Shape: {df.shape}, Name: '{st.session_state.dataframe_name}'")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.session_state.dataframe = None

    if st.session_state.dataframe is not None:
        if st.button("Clear Loaded Data"):
            st.session_state.dataframe = None
            st.session_state.agent = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.header("2. Agent Setup")
    if st.session_state.dataframe and st.session_state.agent is None:
        st.info("Setting up agent...")
        try:
            df_tools = DataFrameTools(
                df=st.session_state.dataframe,
                df_name=st.session_state.dataframe_name,
                max_rows_display=50
            )

            agent = Agent(
                model=OpenAILike(base_url=MODEL_BASE_URL),
                tools=[df_tools],
                markdown=True,
                retries=2,
                show_tool_calls=True,
                instructions=[
                    "You are a helpful data analysis assistant.",
                    f"Use the DataFrame named '{st.session_state.dataframe_name}'.",
                    "Start with 'get_dataframe_info' to understand the data.",
                    "Use tools like 'get_dataframe_head', 'get_dataframe_columns',",
                    "'query_dataframe_sql', and 'execute_python_code' for analysis.",
                    "Use matplotlib, seaborn, or plotly for plots. No plt.show().",
                    "Respond naturally to general questions.",
                    "Explain your answers clearly with examples or tables."
                ],
            )
            st.session_state.agent = agent
            st.success("Agent ready!")
        except Exception as e:
            st.error(f"Failed to init agent: {e}")
            st.session_state.agent = None
    elif st.session_state.agent:
        st.success("Agent is ready.")
    else:
        st.warning("Upload a CSV first.")

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask questions about your data..." if st.session_state.agent else "Upload data to start chat"):
    if st.session_state.agent is None:
        st.warning("Upload a CSV first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""
        try:
            stream = st.session_state.agent.run(prompt, stream=True)
            for chunk in stream:
                if isinstance(chunk, str):
                    response_text += chunk
                elif hasattr(chunk, 'content') and chunk.content:
                    response_text += chunk.content
                placeholder.markdown(response_text + "▌")
            placeholder.markdown(response_text)
        except Exception as e:
            response_text = f"Error: {e}"
            placeholder.error(response_text)
            logger.error(f"Agent error: {e}", exc_info=True)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
