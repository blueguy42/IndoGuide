import streamlit as st
import json
import uuid
from datetime import datetime
from core.llm_client import LLMClient
from core.logger import DialogueLogger
from core.rag_system import RAGSystem
from config.config import MODEL_NAME, RAG_NAME_TO_ID, STARTER_MESSAGE, PERSONAS, DEFAULT_PERSONA, get_prompt


# Initialize logger
logger = DialogueLogger()

# Page configuration
st.set_page_config(
    page_title="IndoGuide",
    page_icon="🇮🇩",
    layout="centered"
)

# Load custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject Google Material Symbols Font
st.markdown(
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&icon_names=chat_add_on" />',
    unsafe_allow_html=True
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.rag_config = 1  # Default to baseline
    st.session_state.rag_system = None  # Will be initialized based on config
    st.session_state.persona = DEFAULT_PERSONA
    st.session_state.session_log = None  # Will be created on first user input
    st.session_state.session_started = False  # Track if user has sent first message
    st.session_state.llm_client = LLMClient(model=MODEL_NAME)
    st.session_state.llm_client.add_assistant_message(STARTER_MESSAGE)


def restart_conversation():
    """Start a new conversation session"""
    # Save current session only if it was started
    if st.session_state.session_started:
        logger.save_session(st.session_state.session_log)
    
    # Create new session
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_log = None  # Will be created on first user input
    st.session_state.session_started = False
    st.session_state.llm_client.reset_conversation()
    st.session_state.llm_client.add_assistant_message(STARTER_MESSAGE)


# Title and session info
st.markdown(
    """
    <div class="info-container">
        <h1 style="margin: 0;">🇮🇩 IndoGuide</h1>
        <div class="info-button-wrapper">
            <span class="info-button">i</span>
            <span class="tooltip-text">IndoGuide is your smart travel companion designed to make exploring Indonesia effortless. Ask away information on must-see destinations, visas, transportation, safety, and local etiquettes, so you can travel with confidence. Whether you're planning your itinerary or navigating on the go, IndoGuide helps you experience Indonesia like a pro!</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption(f"**Session ID:** `{st.session_state.session_id}`")

st.divider()

# Sidebar for Configuration
with st.sidebar:
    st.header("🎭 Persona")
    
    # helper to format display name
    persona_options = list(PERSONAS.keys())
    
    selected_persona = st.radio(
        "Select Persona:",
        options=persona_options,
        format_func=lambda x: PERSONAS[x]["name"],
        index=persona_options.index(st.session_state.persona),
        help="Choose the personality of the assistant."
    )
    
    # Update persona in session state
    if selected_persona != st.session_state.persona:
        st.session_state.persona = selected_persona
        # Restart session when persona changes
        restart_conversation()
        st.rerun()

    st.divider()
    st.header("⚙️ RAG Configuration")
    
    selected_config_name = st.radio(
        "Select RAG Configuration:",
        options=list(RAG_NAME_TO_ID.keys()),
        index=st.session_state.rag_config - 1,
        help="Choose how retrieved context is ranked before being used."
    )
    
    new_config = RAG_NAME_TO_ID[selected_config_name]
    
    # Initialize RAG system if not set (first run)
    if st.session_state.rag_system is None:
        with st.spinner(f"Initializing RAG system..."):
            st.session_state.rag_system = RAGSystem(config=st.session_state.rag_config)

    # Update RAG config if changed
    if new_config != st.session_state.rag_config:
        st.session_state.rag_config = new_config
        with st.spinner(f"Initializing RAG system..."):
            st.session_state.rag_system = RAGSystem(config=new_config)
        restart_conversation()
        st.rerun()
    
    st.divider()
    if st.button("New Chat", icon=":material/chat_add_on:", use_container_width=True):
        restart_conversation()
        st.rerun()

    save_chat_placeholder = st.empty()
    
    # Render initial state immediately to ensure visibility
    with save_chat_placeholder:
        if st.session_state.session_log:
            st.download_button(
                label="Save Chat History",
                data=json.dumps(st.session_state.session_log, indent=2),
                file_name=f"chat_history_{st.session_state.session_id}.json",
                mime="application/json",
                icon=":material/download:", 
                use_container_width=True,
                key="save_chat_init"
            )
        else:
            if st.button("Save Chat History", icon=":material/download:", use_container_width=True, key="save_chat_init_warn"):
                st.warning("No chat history to save yet. Start a conversation first!")

# Display chat messages from LLM client
for message in st.session_state.llm_client.get_messages():
    avatar = "assets/icon_assistant.png" if message["role"] == "assistant" else "assets/icon_user.png"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Create session log on first user input
    if not st.session_state.session_started:
        st.session_state.session_log = logger.create_session(
            st.session_state.session_id, 
            st.session_state.rag_config,
            persona=st.session_state.persona,
            model_name=MODEL_NAME
        )
        st.session_state.session_started = True
    
    # Get user timestamp
    user_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display user message
    with st.chat_message("user", avatar="assets/icon_user.png"):
        st.markdown(prompt)
    
    # Retrieve relevant context using RAG
    with st.spinner("Retrieving relevant information..."):
        retrieved_snippets = st.session_state.rag_system.retrieve(prompt)
        context = st.session_state.rag_system.format_context(retrieved_snippets)
    
    # Inject context into system prompt
    current_prompt_key = PERSONAS[st.session_state.persona]["prompt_key"]
    system_prompt_text = get_prompt(current_prompt_key)
    
    augmented_prompt = context + "\n" + system_prompt_text
    
    # If this is the first real turn (only starter message in history), inform the model
    if len(st.session_state.llm_client.messages) == 1:
        print("Informing model about starter message...")
        augmented_prompt += f"\n\n[Context: You have just started the conversation with this greeting, so do not introduce yourself again: '{STARTER_MESSAGE}']"
    
    # Log user turn with retrieved snippets
    logger.add_turn(
        st.session_state.session_log,
        speaker="user",
        utterance=prompt,
        timestamp=user_timestamp,
        retrieved_snippets=retrieved_snippets
    )
    
    # Get bot response
    with st.chat_message("assistant", avatar="assets/icon_assistant.png"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Show thinking indicator while waiting for response
        with st.spinner("Thinking..."):
            # Stream the response with augmented prompt
            for chunk in st.session_state.llm_client.chat_stream(
                user_message=prompt,
                system_prompt=augmented_prompt
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    
    # Get bot timestamp (when streaming finished)
    bot_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log bot turn
    logger.add_turn(
        st.session_state.session_log,
        speaker="assistant",
        utterance=full_response,
        timestamp=bot_timestamp
    )
    
    # Auto-save session after each turn
    logger.save_session(st.session_state.session_log)



# Update Save Chat button with latest session log if available
if st.session_state.session_log:
    with save_chat_placeholder:
        st.download_button(
            label="Save Chat History",
            data=json.dumps(st.session_state.session_log, indent=2),
            file_name=f"chat_history_{st.session_state.session_id}.json",
            mime="application/json",
            icon=":material/download:", 
            use_container_width=True,
            key="save_chat_final"
        )
