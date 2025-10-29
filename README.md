# Nemotron Farm Planner

This application demonstrates a multi-agent system for farm task planning using **NVIDIA Nemotron**. It allows users to input farm goals via text or audio, which are then processed and executed by a chain of specialized AI agents to generate a validated and summarized farm plan.

---

## 🚀 Features

*   **Multi-Agent Architecture:** A sophisticated system comprising several specialized AI agents working in conjunction.
    *   **Interface Agent:** Processes human input (text or audio), augments it for better prompting, and clarifies ambiguities for the planning agent.
    *   **Planner Agent:** Parses natural-language farm goals into structured tasks, utilizing function calls for reasoning about constraints like distance, time-windows, and skill checks.
    *   **Dispatcher Agent:** Assigns tasks to workers based on proximity, skill, and shift availability, outputting a per-worker task order.
    *   **Safety Agent:** Automatically checks for and inserts safety-related tasks, such as hydration breaks, considering factors like re-entry intervals (REI), personal protective equipment (PPE), and heat index.
    *   **Validator Agent:** Critically evaluates the outputs from the Planner and Dispatcher agents, ensuring the generated plan is feasible, realistic, and free from overlaps or timing conflicts.
    *   **Summarizer Agent:** Generates a concise, human-readable summary of the final validated farm plan.
*   **Flexible Input:** Provide instructions via text input or voice recording using `st.audio_input`.
*   **Dynamic Configuration:** Edit system prompts for each agent directly within the Streamlit UI.
*   **Interactive Data Editors:** Modify worker and task data using `st.data_editor` components.
*   **Visual Task & Worker Locations:** Side-by-side maps display the geographical distribution of workers and tasks.
*   **Detailed Output:** View structured tasks, assignments with integrated water breaks, and notes from the planning process.

---

## ⚙️ Setup and Run

### 1. Requirements
- Python 3.10+
- An NVIDIA API Key

### 2. Environment Variables

Set your NVIDIA API key in a `.env` file or as an environment variable:

```
NIM_KEY="<your_api_key>"
```

### 3. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `app.py` | The main Streamlit application, implementing the multi-agent system and UI. |
| `requirements.txt` | Python dependencies for the project. |
| `.env` | Environment variables, including your NVIDIA API key. |
| `README.md` | This comprehensive guide to the Nemotron Farm Planner. |

---

© 2025 | Built for demonstration purposes
