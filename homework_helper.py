import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

st.set_page_config(
    page_title="AI Homework Helper",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------
# API KEY
# ---------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error(
            "No API key found. Add OPENAI_API_KEY to Streamlit Secrets "
            "or set the OPENAI_API_KEY environment variable."
        )
        st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
llm = LLM(
    model="openai/gpt-4o-mini",
    temperature=0.4,
)

MAX_RUNS = 5


# ---------------------------------------------------------
# AGENTS
# ---------------------------------------------------------
def build_agents():
    teacher = Agent(
        role="Teacher",
        goal="Explain any topic clearly to a 15-year-old.",
        backstory=(
            "You are a patient teacher who explains things using simple "
            "language and everyday examples. You never use a technical word "
            "without explaining it first. You assume the student is clever "
            "but completely new to this topic."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )

    worked_example_writer = Agent(
        role="Worked Example Writer",
        goal=(
            "Create useful practice questions based directly on the "
            "teacher's explanation and solve each question step by step."
        ),
        backstory=(
            "You are a teacher's assistant. Your job is to take the teacher's "
            "explanation and turn it into worked examples that a student can "
            "follow and then use as a model for solving similar questions."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )

    quiz_master = Agent(
        role="Quiz Master",
        goal="Write questions that test real understanding, not just memory.",
        backstory=(
            "You create questions that reveal whether a student actually "
            "understands the topic. Wrong answers should be plausible rather "
            "than silly. You explain why the correct answer is correct."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )

    return teacher, worked_example_writer, quiz_master


# ---------------------------------------------------------
# ONE AGENT = ONE CREW
#
# Each Crew is sequential internally, but contains only
# one task. The important sequential pipeline is below:
#
# Teacher → Worked Examples → Quiz Master
#
# The output of each agent is explicitly passed to the
# next agent. This makes the dependency unambiguous.
# ---------------------------------------------------------
def run_agent(agent, description, expected_output):
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    return str(crew.kickoff())


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "runs" not in st.session_state:
    st.session_state.runs = 0

if "pack" not in st.session_state:
    st.session_state.pack = None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 👥 Your crew")

    st.markdown(
        "**🧠 Teacher** — explains the topic\n\n"
        "**📝 Worked Example Writer** — creates and solves examples\n\n"
        "**🎯 Quiz Master** — tests your understanding"
    )

    st.divider()

    level = st.select_slider(
        "📚 Level",
        options=["Beginner", "O Level", "A Level"],
        value="O Level",
    )

    n_questions = st.slider(
        "🎯 Quiz questions",
        min_value=3,
        max_value=8,
        value=5,
    )

    st.divider()

    runs_left = max(0, MAX_RUNS - st.session_state.runs)
    st.markdown(f"**Runs left:** {runs_left}")

    if st.session_state.pack is not None:
        if st.button("🔄 Start over", use_container_width=True):
            st.session_state.pack = None
            st.rerun()


# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------
st.title("🧠 AI Homework Helper")
st.caption(
    "Three AI agents work sequentially to explain, demonstrate, "
    "and test any topic."
)

topic = st.text_input(
    "What do you want help with?",
    placeholder=(
        "e.g. quadratic equations, photosynthesis, "
        "Newton's laws, the French Revolution"
    ),
    max_chars=120,
)


# ---------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------
if st.button(
    "🚀 Build my study pack",
    type="primary",
    use_container_width=True,
):
    if st.session_state.runs >= MAX_RUNS:
        st.error(
            "You have used all your runs. Refresh the page to reset."
        )

    elif len(topic.strip()) < 3:
        st.warning("Type a topic first.")

    else:
        selected_topic = topic.strip()

        try:
            teacher, worked_example_writer, quiz_master = build_agents()

            # =================================================
            # AGENT 1: TEACHER
            # =================================================
            with st.status(
                "🧠 Teacher is explaining the topic...",
                expanded=True,
            ) as status:

                explanation = run_agent(
                    teacher,
                    (
                        f"Explain '{selected_topic}' to a {level} student. "
                        "Use simple language and at least one everyday "
                        "example. Introduce important terminology clearly. "
                        "Keep the explanation under 300 words."
                    ),
                    (
                        "A clear explanation of the topic suitable for "
                        f"a {level} student, including at least one "
                        "real-world example."
                    ),
                )

                status.update(
                    label="🧠 Teacher has finished",
                    state="complete",
                )

            # =================================================
            # AGENT 2: WORKED EXAMPLES
            #
            # This agent receives the actual output of Agent 1.
            # =================================================
            with st.status(
                "📝 Worked Example Writer is creating examples...",
                expanded=True,
            ) as status:

                worked_examples = run_agent(
                    worked_example_writer,
                    (
                        f"You are creating worked examples for a {level} "
                        f"student studying '{selected_topic}'.\n\n"
                        "Here is the explanation produced by the Teacher:\n\n"
                        f"{explanation}\n\n"
                        "Using ONLY the concepts covered in that "
                        "explanation, create 3 practice questions. "
                        "For each question:\n"
                        "1. State the question clearly.\n"
                        "2. Show the solution step by step.\n"
                        "3. Briefly explain why each important step was taken.\n"
                        "4. Make the examples progressively more difficult.\n\n"
                        "The goal is to teach the student how to solve "
                        "similar questions themselves."
                    ),
                    (
                        "Three progressively difficult practice questions "
                        "with complete step-by-step solutions and short "
                        "explanations of the reasoning."
                    ),
                )

                status.update(
                    label="📝 Worked examples are ready",
                    state="complete",
                )

            # =================================================
            # AGENT 3: QUIZ MASTER
            #
            # This agent receives BOTH previous outputs.
            # =================================================
            with st.status(
                "🎯 Quiz Master is testing your understanding...",
                expanded=True,
            ) as status:

                quiz = run_agent(
                    quiz_master,
                    (
                        f"Create {n_questions} multiple-choice questions "
                        f"for a {level} student studying "
                        f"'{selected_topic}'.\n\n"
                        "Teacher explanation:\n"
                        f"{explanation}\n\n"
                        "Worked examples:\n"
                        f"{worked_examples}\n\n"
                        "Use these materials to test whether the student "
                        "understands the concepts and can apply them. "
                        "Do not simply copy the worked-example questions. "
                        "Make the wrong options plausible.\n\n"
                        "After the questions, provide an ANSWERS section "
                        "with the correct answer and a short explanation "
                        "for every question."
                    ),
                    (
                        f"{n_questions} multiple-choice questions with "
                        "options A-D, followed by an ANSWERS section "
                        "explaining every correct answer."
                    ),
                )

                status.update(
                    label="🎯 Quiz is ready",
                    state="complete",
                )

            # Only count a run after all three agents succeed.
            st.session_state.runs += 1

            st.session_state.pack = {
                "topic": selected_topic,
                "level": level,
                "explanation": explanation,
                "worked_examples": worked_examples,
                "quiz": quiz,
            }

            st.rerun()

        except Exception as e:
            st.error("Something went wrong while building the study pack.")
            st.exception(e)


# ---------------------------------------------------------
# DISPLAY RESULT
# ---------------------------------------------------------
pack = st.session_state.pack

if pack is not None:
    st.divider()

    st.subheader(
        f"📦 Study pack: {pack['topic']} · {pack['level']}"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "🧠 Explanation",
            "📝 Worked Examples",
            "🎯 Quiz",
        ]
    )

    with tab1:
        st.markdown(pack["explanation"])

    with tab2:
        st.markdown(pack["worked_examples"])

    with tab3:
        st.markdown(pack["quiz"])

    download_text = (
        f"STUDY PACK: {pack['topic']} ({pack['level']})\n\n"
        "=== EXPLANATION ===\n"
        f"{pack['explanation']}\n\n"
        "=== WORKED EXAMPLES ===\n"
        f"{pack['worked_examples']}\n\n"
        "=== QUIZ ===\n"
        f"{pack['quiz']}"
    )

    st.download_button(
        "⬇️ Download the whole pack",
        data=download_text,
        file_name=(
            f"study_pack_"
            f"{pack['topic'][:20].replace(' ', '_')}.txt"
        ),
        mime="text/plain",
    )
