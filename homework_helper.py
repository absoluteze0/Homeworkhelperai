import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

st.set_page_config(page_title="AI homework helper", page_icon="🧠", layout="wide")

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("No API key found. Add OPENAI_API_KEY in Settings -> Secrets.")
    st.stop()

llm = LLM(model="openai/gpt-4o-mini", temperature=0.4)

MAX_RUNS = 5


def build_agents():
    Teacher = Agent(
        role="Teacher",
        goal="Explain any topic clearly to a 15-year-old",
        backstory=(
            "You are a patient teacher who explains things using simple "
            "language and everyday examples. You never use a technical word "
            "without explaining it first. You assume the student is clever "
            "but completely new to this topic."
        ),
        llm=llm, verbose=False, allow_delegation=False, max_iter=3,
    )
    note_taker = Agent(
        role="Worked Example Writer",
        goal="Write practice questions according to the topic explained by the Explainer and solving them step by step.",
        backstory=(
            "You are a Teachers Assistant tasked with creating example "
            "Questions for the students and solving them step by step "
            "so that the students can solve questions themselves."
        ),
        llm=llm, verbose=False, allow_delegation=False, max_iter=3,
    )
    quiz_master = Agent(
        role="Quiz Master",
        goal="Write questions that test real understanding, not memory",
        backstory=(
            "You write quiz questions that catch students who have memorised "
            "words without understanding them. Your wrong answers are always "
            "plausible, never silly. You always explain why the correct "
            "answer is correct."
        ),
        llm=llm, verbose=False, allow_delegation=False, max_iter=3,
    )
    return Teacher, note_taker, quiz_master


def run_one(agent, description, expected_output):
    task = Task(description=description, expected_output=expected_output,
                agent=agent)
    return str(Crew(agents=[agent], tasks=[task],
                    process=Process.sequential, verbose=False).kickoff())


for k, v in [("runs", 0), ("pack", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.markdown("### 👥 Your crew")
    st.markdown("**🧠 Teacher** — explains it\n\n"
                "**📝 Notes Specialist** — condenses it\n\n"
                "**🎯 Quiz Master** — tests you")
    st.divider()
    level = st.select_slider("📚 Level",
                             options=["Beginner", "O Level", "A Level"],
                             value="O Level")
    n_questions = st.slider("🎯 Quiz questions", 3, 8, 5)
    st.divider()
    st.markdown(f"**Runs left:** {MAX_RUNS - st.session_state.runs}")
    if st.session_state.pack and st.button("🔄 Start over",
                                           use_container_width=True):
        st.session_state.pack = None
        st.rerun()

st.title("🧠 AI Study Crew")
st.caption("Three AI agents build you a complete study pack on any topic.")

topic = st.text_input(
    "What do you want to study?",
    placeholder="e.g. photosynthesis, the French Revolution, quadratic equations",
    max_chars=120,
)

if st.button("🚀 Build my study pack", type="primary",
             use_container_width=True):
    if st.session_state.runs >= MAX_RUNS:
        st.error("You have used all your runs. Refresh the page to reset.")
    elif len(topic.strip()) < 3:
        st.warning("Type a topic first.")
    else:
        t = topic.strip()
        try:
            teacher, note_taker, quiz_master = build_agents()

            with st.status("🧠 Teacher is explaining...") as s:
                explanation = run_one(
                    teacher,
                    f"Explain '{t}' to a {level} student. Use simple language "
                    f"and at least one everyday example. Under 300 words.",
                    "A clear explanation with a real-world example.",
                )
                s.update(label="🧠 Teacher has explained it", state="complete")

            with st.status("📝 Notes Specialist is condensing...") as s:
                notes = run_one(
                    note_taker,
                    f"Turn this explanation into revision notes for a {level} "
                    f"student:\n\n{explanation}",
                    "6-8 short bullet points. Bold the key terms.",
                )
                s.update(label="📝 Notes are ready", state="complete")

            with st.status("🎯 Quiz Master is writing questions...") as s:
                quiz = run_one(
                    quiz_master,
                    f"Write {n_questions} multiple-choice questions on '{t}' "
                    f"for a {level} student, based on these notes:\n\n{notes}",
                    f"{n_questions} questions with options A-D, then an "
                    "ANSWERS section explaining each correct answer.",
                )
                s.update(label="🎯 Quiz is ready", state="complete")

            st.session_state.runs += 1
            st.session_state.pack = {
                "topic": t, "level": level, "explanation": explanation,
                "notes": notes, "quiz": quiz,
            }
            st.rerun()

        except Exception as e:
            st.error("Something went wrong.")
            st.caption(f"{type(e).__name__}: {e}")

p = st.session_state.pack
if p:
    st.divider()
    st.subheader(f"📦 Study pack: {p['topic']} · {p['level']}")

    tab1, tab2, tab3 = st.tabs(["🧠 Explanation", "📝 Notes", "🎯 Quiz"])
    with tab1:
        st.markdown(p["explanation"])
    with tab2:
        st.markdown(p["notes"])
    with tab3:
        st.markdown(p["quiz"])

    st.download_button(
        "⬇️ Download the whole pack",
        data=(f"STUDY PACK: {p['topic']} ({p['level']})\n\n"
              f"=== EXPLANATION ===\n{p['explanation']}\n\n"
              f"=== NOTES ===\n{p['notes']}\n\n"
              f"=== QUIZ ===\n{p['quiz']}"),
        file_name=f"study_pack_{p['topic'][:20].replace(' ', '_')}.txt",
        mime="text/plain",
    )
