import os
import streamlit as st

def update_env_file(key_values: dict):
    """
    Saves or updates specific key-value pairs in the local .env file.
    Preserves existing lines and comments to prevent data loss.
    """
    env_path = ".env"
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
    updated_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if "=" in stripped:
            k, v = stripped.split("=", 1)
            k = k.strip()
            if k in key_values:
                new_lines.append(f"{k}={key_values[k]}\n")
                updated_keys.add(k)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    for k, v in key_values.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}\n")
            
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

# Page Layout & Header
st.title("⚙️ System Credentials & Settings")
st.markdown("Configure global API credentials for the remote Kaggle cleaning execution and Groq LLM profiling.")

# Load current keys from state or environment variables
current_kaggle_user = st.session_state.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME", "")
current_kaggle_key = st.session_state.get("KAGGLE_KEY") or os.environ.get("KAGGLE_KEY", "")
current_groq_key = st.session_state.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")

with st.form("settings_form", clear_on_submit=False):
    st.subheader("Remote Service Credentials")
    
    kaggle_user = st.text_input(
        "Kaggle Username", 
        value=current_kaggle_user,
        help="Your Kaggle account username (retrieved from Kaggle Account API settings)."
    )
    
    kaggle_key = st.text_input(
        "Kaggle API Key", 
        value=current_kaggle_key,
        type="password",
        help="Your Kaggle API Token (generated as a kaggle.json token)."
    )
    
    groq_key = st.text_input(
        "Groq API Key", 
        value=current_groq_key,
        type="password",
        help="API Key for Groq LLM execution to profile dataset and build cleaning strategies."
    )
    
    submit_btn = st.form_submit_button("Save Credentials", type="primary")

if submit_btn:
    if not kaggle_user.strip() or not kaggle_key.strip() or not groq_key.strip():
        st.error("All credentials must be provided to run remote workloads.")
    else:
        # Update Session State
        st.session_state["KAGGLE_USERNAME"] = kaggle_user.strip()
        st.session_state["KAGGLE_KEY"] = kaggle_key.strip()
        st.session_state["GROQ_API_KEY"] = groq_key.strip()
        
        # Update Environment Variables for active session
        os.environ["KAGGLE_USERNAME"] = kaggle_user.strip()
        os.environ["KAGGLE_KEY"] = kaggle_key.strip()
        os.environ["GROQ_API_KEY"] = groq_key.strip()
        
        # Persist locally in .env file
        try:
            update_env_file({
                "KAGGLE_USERNAME": kaggle_user.strip(),
                "KAGGLE_KEY": kaggle_key.strip(),
                "GROQ_API_KEY": groq_key.strip()
            })
            st.success("Credentials saved and persisted successfully!")
        except Exception as e:
            st.error(f"Failed to persist credentials to .env file: {e}")
