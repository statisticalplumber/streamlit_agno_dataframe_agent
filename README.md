# 📊 Smart DataFrame Agent App – Chat with Your Data

Agent LLM is an interactive, AI-powered data exploration and analysis tool built with [Streamlit](https://streamlit.io/) and [Agno](https://github.com/agno-ai). Upload a CSV file and start a conversation with your data using natural language. The app leverages a local or remote LLM backend to analyze, query, and visualize data intelligently.

---

## 🚀 Features

### ✅ Chat-Driven Data Exploration
- Upload a CSV file and begin querying your data conversationally.
- Ask general questions or dive deep into data trends, summaries, and insights.
- Built-in agent reasoning using Agno’s tool-based framework.

### 📊 DataFrame Analysis Tools
- Automatically integrates your dataset with intelligent tools:
  - `get_dataframe_info`: Understand structure, column types, and more.
  - `get_dataframe_columns`: Quick access to all column names.
  - `get_dataframe_head`: Preview the first few rows.
  - `query_dataframe_sql`: Use SQL to explore your DataFrame.
  - `execute_python_code`: Write custom Python code using pandas, matplotlib, seaborn, or plotly.

### 📈 Built-in Plotting Support
- Supports visualizations with:
  - **Matplotlib**
  - **Seaborn**
  - **Plotly**
- Automatically renders generated plots in chat responses.

### 🧠 Local or Cloud LLM Support
- Compatible with local LLMs (e.g., via LM Studio) or OpenAI-compatible APIs.
- Easy configuration of model endpoint and API key.

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- [Streamlit](https://docs.streamlit.io/)
- Local or cloud-based LLM endpoint (e.g., LM Studio, Ollama, OpenAI)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/agent-llm-data-chat.git
cd agent-llm-data-chat
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the app
```bash
streamlit run app.py
```

---

## ⚙️ Configuration

Modify the following section in `app.py` to configure your model:

```python
MODEL_BASE_URL = "http://localhost:1234/v1"  # Change this if your model runs elsewhere
MODEL_API_KEY = "lm-studio"                  # Optional, depending on your model
```

---

## 📁 Project Structure

```
agent-llm-data-chat/
│
├── app.py                 # Main Streamlit application
├── dataframe_tools.py     # Custom tools to explore and analyze the DataFrame
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🧠 Powered By

- [Agno](https://github.com/agno-ai) – Framework for building agentic LLM applications.
- [Pandas](https://pandas.pydata.org/) – DataFrame manipulation.
- [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Plotly](https://plotly.com/python/) – Visualization libraries.
- [Streamlit](https://streamlit.io/) – Rapid app development framework.

---

## 🛠️ TODOs / Future Enhancements

- [ ] Multi-file data comparison
- [ ] Agent memory support
- [ ] Model selection in UI
- [ ] Caching for large datasets

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues, submit PRs, or share ideas for improvement.

---

## 📬 Contact

For questions or feedback, please reach out or open an issue on the [GitHub repository](https://github.com/your-username/agent-llm-data-chat).
