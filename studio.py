import os
import requests
import streamlit as st

# Helper function to persist credentials inside dialog submissions
def persist_credentials(kaggle_user, kaggle_key, groq_key):
    """Saves keys in session_state, current process env, and updates .env file."""
    st.session_state["KAGGLE_USERNAME"] = kaggle_user
    st.session_state["KAGGLE_KEY"] = kaggle_key
    st.session_state["GROQ_API_KEY"] = groq_key
    
    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"] = kaggle_key
    os.environ["GROQ_API_KEY"] = groq_key
    
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
            if k in ["KAGGLE_USERNAME", "KAGGLE_KEY", "GROQ_API_KEY"]:
                val = kaggle_user if k == "KAGGLE_USERNAME" else (kaggle_key if k == "KAGGLE_KEY" else groq_key)
                new_lines.append(f"{k}={val}\n")
                updated_keys.add(k)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    for k, v in [("KAGGLE_USERNAME", kaggle_user), ("KAGGLE_KEY", kaggle_key), ("GROQ_API_KEY", groq_key)]:
        if k not in updated_keys:
            new_lines.append(f"{k}={v}\n")
            
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


# 1. Native Streamlit Dialog Modal definition
@st.dialog("API Credentials Required")
def require_credentials_dialog():
    """
    Renders a secure credential acquisition form.
    Blocks user flow until credentials are set, and re-runs the page.
    """
    st.warning("To run the Data Cleaning Studio on remote Kaggle servers, you must provide your Kaggle credentials and a Groq API key.")
    
    user = st.text_input(
        "Kaggle Username", 
        value=st.session_state.get("KAGGLE_USERNAME", os.environ.get("KAGGLE_USERNAME", ""))
    )
    key = st.text_input(
        "Kaggle API Key", 
        value=st.session_state.get("KAGGLE_KEY", os.environ.get("KAGGLE_KEY", "")), 
        type="password"
    )
    groq = st.text_input(
        "Groq API Key", 
        value=st.session_state.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")), 
        type="password"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save & Proceed", type="primary", use_container_width=True):
            if not user.strip() or not key.strip() or not groq.strip():
                st.error("Please fill in all credential fields.")
            else:
                # Save, write env, and reload state
                persist_credentials(user.strip(), key.strip(), groq.strip())
                st.success("Credentials saved! Starting execution...")
                st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# 2. Studio Main UI & Execution Hook
def main_studio_page():
    st.title("🧼 Al-Dalil Data Cleaning Studio")
    st.subheader("Data Cleaning Controls")
    
    # Check credentials in session state or host environment
    kaggle_user = st.session_state.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME")
    kaggle_key = st.session_state.get("KAGGLE_KEY") or os.environ.get("KAGGLE_KEY")
    groq_key = st.session_state.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    
    dataset_id = st.session_state.get("active_dataset_id", "data_001")
    strategy = st.selectbox("Cleaning Strategy", ["alpha", "beta", "gamma"])
    goal = st.text_input("Custom Goal / Directive (Optional)")

    # Execution Hook trigger button
    if st.button("🚀 Clean on Kaggle", type="primary"):
        if not kaggle_user or not kaggle_key or not groq_key:
            # Trigger Dialog modal overlay if keys are missing
            require_credentials_dialog()
        else:
            # Keys verified, launch Remote Kaggle workload
            with st.spinner("Pushing workload and initializing Kaggle session..."):
                try:
                    # Request to local API server which handles orchestration
                    response = requests.post(
                        "http://127.0.0.1:8000/api/clean",
                        data={
                            "dataset_id": dataset_id,
                            "strategy": strategy,
                            "goal": goal
                        }
                    )
                    if response.status_code in [200, 202]:
                        st.success(f"Workload successfully offloaded to Kaggle! Task ID: {response.json().get('task_id')}")
                    else:
                        st.error(f"Failed to submit task: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend orchestration server: {e}")

if __name__ == "__main__":
    main_studio_page()
