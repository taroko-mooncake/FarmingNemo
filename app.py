import os, json, re
from math import radians, sin, cos, asin, sqrt
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

# Optional tolerant parser
try:
    import json5
    USE_JSON5 = True
except ImportError:
    USE_JSON5 = False

# =========================================================
# Environment setup
# =========================================================
load_dotenv()
NIM_KEY = os.getenv("NIM_KEY")
NIM_MODEL = os.getenv("NIM_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

if not NIM_KEY:
    st.error("Missing NIM_KEY in .env file")
    st.stop()

client = OpenAI(base_url=NIM_BASE_URL, api_key=NIM_KEY)

# =========================================================
# Utility functions
# =========================================================
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def fits_time_window(start, end, earliest, latest, duration_min):
    hhmm = lambda s: datetime.strptime(s, "%H:%M")
    s, e = hhmm(start), hhmm(end)
    earliest_t, latest_t = hhmm(earliest), hhmm(latest)
    finish = s + timedelta(minutes=int(duration_min))
    return (
        s.time() >= earliest_t.time()
        and finish.time() <= latest_t.time()
        and finish.time() <= e.time()
    )

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "haversine_m",
            "description": "Compute great-circle distance (m).",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat1": {"type": "number"},
                    "lon1": {"type": "number"},
                    "lat2": {"type": "number"},
                    "lon2": {"type": "number"},
                },
                "required": ["lat1", "lon1", "lat2", "lon2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fits_time_window",
            "description": "True if a task fits within worker schedule and task window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "earliest": {"type": "string"},
                    "latest": {"type": "string"},
                    "duration_min": {"type": "integer"},
                },
                "required": ["start", "end", "earliest", "latest", "duration_min"],
            },
        },
    },
]

# =========================================================
# System prompt (strict JSON)
# =========================================================
PLANNER_PROMPT = """You are a farm labor coordinator.

You must respond with ONE valid JSON object and nothing else.
Do NOT include explanations, introductions, markdown, or text outside JSON.

Strictly follow this schema:
{
 "tasks": [...],
 "assignments": [
    {
        "worker": str,
        "tasks": list[str],
        "water_breaks": list[str]
    }
 ],
 "notes": [...] 
}

Rules:
- Output ONLY valid JSON.
- Prefer higher priority (1 = highest).
- Match worker skills exactly.
- Use provided tool functions if needed.
"""

INTERFACE_PROMPT = """You are an interface agent that helps a planning agent understand user instructions."""

INSTRUCTION_PROCESSING_SYSTEM_PROMPT = """You are an expert in understanding and clarifying instructions. 
Your task is to process the user's instruction and make it clear and concise for a planning agent. 
Focus on extracting the key information and constraints.
Respond with only the processed instruction.
"""

# =========================================================
# JSON extraction helper
# =========================================================
def extract_json_objects(text: str):
    objs = []
    i, n = 0, len(text)
    brace_stack = []
    in_str = False
    esc = False
    start_idx = None
    while i < n:
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                if not brace_stack:
                    start_idx = i
                brace_stack.append("{")
            elif ch == "}":
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start_idx is not None:
                        objs.append(text[start_idx:i+1])
                        start_idx = None
        i += 1
    return objs

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Farming Nemo", layout="wide")
st.title("🌾 Farming Nemo")
st.text("Outdoor farm work is hard to coordinate — plans change fast, and manual scheduling wastes time and causes errors. \
This app uses agentic AI with NVIDIA Nemotron to turn plain-language farm instructions into optimized, safe task assignments,\
helping crews work smarter and adapt instantly in the field.")
st.markdown("Real-time reasoning and structured farm task planning using **NVIDIA Nemotron**.")

default_instruction = (
    "Harvest the lettuce, prune the tomatoes and irrigate everything south of latitude 36.55. "
    "Alpha can't travel more then 2 miles; finish by 3pm; add 10-min hydration breaks every 90 min."
)

st.subheader('Input Instructions and System Prompts')
st.info(
        "**Worker JSON fields:** `name`, `skill` (e.g., `harvest_l1`, `irrigation_l2`), "
        "`start`/`end` (HH:MM), `lat`/`lon` (decimal degrees).\n\n"
        "**Task JSON fields:** `id`, `type` (e.g., `harvest`, `weeding`), `species`, `block`, `row`, `lat`/`lon`, "
        "`duration_min`, `skill`, `earliest`/`latest` (HH:MM), `priority` (1-5).",
    )
col1, col2, col3 = st.columns(3)
with col1:
    st.text_area("planner prompt", INSTRUCTION_PROCESSING_SYSTEM_PROMPT, height=150)
with col2:
    st.text_area("user prompt", PLANNER_PROMPT, height=150)
with col3:
    st.text_area("interface prompt", INTERFACE_PROMPT, height=150)

st.subheader('Farm Manager Daily Instructions')
col1, col2 = st.columns(2)
with col1:
    instruction = st.text_area("Instruction", default_instruction, height=150)
with col2:
    audio_bytes = st.audio_input("Or record your instruction:")

# Default workers and tasks (Salinas Valley, CA)
sample_workers = [
    {"name": "Alpha", "skill": "harvest_l1", "start": "06:30", "end": "16:00", "lat": 36.377, "lon": -121.455},
    {"name": "Beta", "skill": "harvest_l2", "start": "06:30", "end": "17:30", "lat": 36.580, "lon": -121.460},
    {"name": "Gamma", "skill": "prune_l1", "start": "07:00", "end": "15:00", "lat": 36.970, "lon": -121.750},
    {"name": "Delta", "skill": "irrigation_l1", "start": "05:00", "end": "13:00", "lat": 36.490, "lon": -121.480},
    {"name": "Epsilon", "skill": "pest_control_l2", "start": "08:00", "end": "16:00", "lat": 36.860, "lon": -121.440},
    {"name": "Zeta", "skill": "planting_l1", "start": "07:00", "end": "14:30", "lat": 36.552, "lon": -121.443},
    {"name": "Eta", "skill": "weeding_l1", "start": "08:00", "end": "17:00", "lat": 36.563, "lon": -121.447},
    {"name": "Theta", "skill": "harvest_l2", "start": "06:30", "end": "17:00", "lat": 36.553, "lon": -121.565},
    {"name": "Iota", "skill": "irrigation_l1", "start": "05:30", "end": "12:00", "lat": 36.486, "lon": -121.475},
    {"name": "Kappa", "skill": "pest_control_l2", "start": "08:00", "end": "15:30", "lat": 36.461, "lon": -121.138},
    {"name": "Lambda", "skill": "prune_l1", "start": "07:00", "end": "15:00", "lat": 36.272, "lon": -121.449},
    {"name": "Mu", "skill": "harvest_l1", "start": "06:00", "end": "14:00", "lat": 36.575, "lon": -121.758},
    {"name": "Nu", "skill": "planting_l1", "start": "06:30", "end": "13:30", "lat": 36.556, "lon": -121.432},
    {"name": "Xi", "skill": "weeding_l1", "start": "07:30", "end": "16:30", "lat": 36.565, "lon": -121.439},
    {"name": "Omicron", "skill": "harvest_l2", "start": "06:00", "end": "15:00", "lat": 36.579, "lon": -121.462},
]
sample_tasks = [
    {"id": "Field 03", "type": "harvest", "species": "lettuce", "block": "C", "row": "12", "lat": 36.5775, "lon": -121.558,
     "duration_min": 90, "skill": "harvest_l1", "earliest": "06:30", "latest": "17:30", "priority": 2},
    {"id": "Field 02", "type": "prune", "species": "tomatoes", "block": "E", "row": "7", "lat": 36.5782, "lon": -121.562,
     "duration_min": 60, "skill": "prune_l1", "earliest": "06:30", "latest": "17:30", "priority": 3},
    {"id": "Field 01", "type": "harvest", "species": "lettuce", "block": "C", "row": "13", "lat": 36.5572, "lon": -121.452,
     "duration_min": 120, "skill": "harvest_l2", "earliest": "06:30", "latest": "17:30", "priority": 1},
    {"id": "Field 11", "type": "irrigation", "species": "all", "block": "A", "row": "1-5", "lat": 36.595, "lon": -121.485,
     "duration_min": 180, "skill": "irrigation_l1", "earliest": "05:00", "latest": "10:00", "priority": 1},
    {"id": "Field 24", "type": "planting", "species": "corn", "block": "F", "row": "1-20", "lat": 36.555, "lon": -121.435,
     "duration_min": 240, "skill": "planting_l1", "earliest": "07:00", "latest": "12:00", "priority": 2},
    {"id": "Field 25", "type": "weeding", "species": "all", "block": "B", "row": "all", "lat": 36.565, "lon": -121.445,
     "duration_min": 180, "skill": "weeding_l1", "earliest": "08:00", "latest": "17:00", "priority": 4},
    {"id": "Field 09", "type": "pest_control", "species": "all", "block": "D", "row": "all", "lat": 36.588, "lon": -121.470,
     "duration_min": 120, "skill": "pest_control_l2", "earliest": "08:00", "latest": "11:00", "priority": 1},

    # --- new 10 tasks below ---
    {"id": "Field 12", "type": "harvest", "species": "spinach", "block": "B", "row": "6", "lat": 36.179, "lon": -121.666,
     "duration_min": 100, "skill": "harvest_l1", "earliest": "06:30", "latest": "16:30", "priority": 2},
    {"id": "Field 13", "type": "planting", "species": "broccoli", "block": "G", "row": "5-10", "lat": 36.159, "lon": -121.440,
     "duration_min": 150, "skill": "planting_l1", "earliest": "07:00", "latest": "13:00", "priority": 3},
    {"id": "Field 14", "type": "irrigation", "species": "peppers", "block": "A", "row": "2-4", "lat": 36.292, "lon": -121.482,
     "duration_min": 90, "skill": "irrigation_l1", "earliest": "05:00", "latest": "11:00", "priority": 1},
    {"id": "Field 15", "type": "weeding", "species": "all", "block": "C", "row": "8-10", "lat": 36.564, "lon": -121.448,
     "duration_min": 120, "skill": "weeding_l1", "earliest": "08:00", "latest": "17:00", "priority": 3},
    {"id": "Field 16", "type": "pest_control", "species": "strawberries", "block": "H", "row": "3", "lat": 36.386, "lon": -121.472,
     "duration_min": 110, "skill": "pest_control_l2", "earliest": "08:00", "latest": "12:00", "priority": 2},
    {"id": "Field 17", "type": "harvest", "species": "strawberries", "block": "H", "row": "4-5", "lat": 36.485, "lon": -121.474,
     "duration_min": 150, "skill": "harvest_l2", "earliest": "06:30", "latest": "17:30", "priority": 1},
    {"id": "Field 18", "type": "planting", "species": "tomatoes", "block": "G", "row": "1-3", "lat": 36.953, "lon": -121.437,
     "duration_min": 180, "skill": "planting_l1", "earliest": "07:00", "latest": "14:00", "priority": 2},
    {"id": "Field 19", "type": "harvest", "species": "broccoli", "block": "F", "row": "8", "lat": 36.557, "lon": -121.441,
     "duration_min": 120, "skill": "harvest_l1", "earliest": "06:00", "latest": "16:00", "priority": 2},
    {"id": "Field 20", "type": "irrigation", "species": "corn", "block": "A", "row": "6-8", "lat": 36.594, "lon": -121.486,
     "duration_min": 100, "skill": "irrigation_l1", "earliest": "05:00", "latest": "09:30", "priority": 1},
    {"id": "Field 21", "type": "weeding", "species": "lettuce", "block": "B", "row": "9-12", "lat": 36.563, "lon": -121.447,
     "duration_min": 150, "skill": "weeding_l1", "earliest": "08:00", "latest": "17:00", "priority": 4},
]

# Text area input
col1, col2 = st.columns(2)
workers_json = col1.text_area("Workers (JSON)", json.dumps(sample_workers, indent=2), height=250)
tasks_json = col2.text_area("Tasks (JSON)", json.dumps(sample_tasks, indent=2), height=250)

# Parse and preview as tables
try:
    workers_obj = json.loads(workers_json)
    tasks_obj = json.loads(tasks_json)
except Exception as e:
    st.error(f"Invalid JSON format: {e}")
    st.stop()

workers_df_edited = st.data_editor(workers_obj, width='stretch')
tasks_df_edited = st.data_editor(tasks_obj, width='stretch')

workers_df = pd.DataFrame(workers_df_edited)
tasks_df = pd.DataFrame(tasks_df_edited)

map_col1, map_col2 = st.columns(2)
with map_col1:
    st.subheader("👷 Worker Locations")
    st.map(workers_df)
with map_col2:
    st.subheader("🧺 Task Locations")
    st.map(tasks_df)


if st.button("🚜 Plan with Nemotron"):
    if audio_bytes:
        with st.spinner("Transcribing audio..."):
            transcription = client.audio.transcriptions.create(
                model="Parakeet-1.1b-ctc-en-us-asr-set-6.0",
                file=audio_bytes
            )
            instruction = transcription.text
        with st.spinner("Processing instruction..."):
            response = client.chat.completions.create(
                model=NIM_MODEL,
                messages=[
                    {"role": "system", "content": INSTRUCTION_PROCESSING_SYSTEM_PROMPT},
                    {"role": "user", "content": instruction}
                ],
                temperature=0.2,
                top_p=0.9,
            )
            instruction = response.choices[0].message.content

    # Handle both DataFrame and list outputs from st.data_editor
    workers_obj = workers_df.to_dict("records") if hasattr(workers_df, "to_dict") else workers_df
    tasks_obj = tasks_df.to_dict("records") if hasattr(tasks_df, "to_dict") else tasks_df

    msgs = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": "Your response must be raw JSON only, no introductions."},
        {"role": "user", "content": f"Instruction:\n{instruction}\n\nWorkers JSON:\n{json.dumps(workers_obj)}\n\nTasks JSON:\n{json.dumps(tasks_obj)}"},
    ]

    st.info("Streaming Nemotron reasoning...")

    reasoning_box = st.empty()
    collected_text = []
    plan_json = None

    completion = client.chat.completions.create(
        model=NIM_MODEL,
        messages=msgs,
        temperature=0.2,
        top_p=0.9,
        max_tokens=2048,
        extra_body={"min_thinking_tokens": 512, "max_thinking_tokens": 2048},
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
    )

    reasoning_log = ""
    for chunk in completion:
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_log += delta.reasoning_content
            reasoning_box.markdown(f"🧠 **Reasoning:**\n\n{reasoning_log}")
        if delta.content:
            reasoning_log += delta.content
            collected_text.append(delta.content)
            reasoning_box.markdown(f"🧠 **Reasoning:**\n\n{reasoning_log}")

    # Combine output text
    full_text = "".join(collected_text).strip()
    parser = json5 if USE_JSON5 else json

    # Try parsing all balanced objects
    for raw in reversed(extract_json_objects(full_text)):
        try:
            obj = parser.loads(raw)
            if isinstance(obj, dict) and {"tasks", "assignments", "notes"} <= set(obj.keys()):
                plan_json = obj
                break
        except Exception:
            continue

    # Regex fallback
    if not plan_json:
        match = re.search(r"\{[\s\S]*\}", full_text)
        if match:
            try:
                plan_json = parser.loads(match.group(0))
            except Exception:
                pass

    # Display results
    if plan_json:
        st.success("✅ Plan generated successfully!")
        
        assignments = plan_json.get("assignments", [])
        tasks_map = {task['id']: task for task in tasks_obj}
        
        processed_assignments = []
        for assignment in assignments:
            for task_id in assignment.get("tasks", []):
                task = tasks_map.get(task_id)
                if task:
                    processed_assignments.append({
                        "Worker": assignment.get("worker"),
                        "Task ID": task_id,
                        "Type": task.get("type"),
                        "Species": task.get("species"),
                        "Block": task.get("block"),
                        "Row": task.get("row"),
                        "Duration (min)": task.get("duration_min"),
                        "Water Breaks": ", ".join(assignment.get("water_breaks", [])),
                    })

        st.subheader("Assignments")
        st.dataframe(processed_assignments)

        st.subheader("Notes")
        for note in plan_json.get("notes", []):
            st.markdown(f"- {note}")
    else:
        st.warning("⚠️ No valid JSON plan found in model output.")
        st.text_area("Raw Output", full_text[:3000], height=400)
