import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory

# Define calming prompts for each state
calm_prompts = {
    "depression": """
        You are a calming and empathetic chatbot. If a user is feeling depressed, respond with gentle, positive, and encouraging messages. Help them focus on hope, self-care, and positivity.
        Example:
        - "I'm here to listen. You are stronger than you think."
        - "Sometimes, small steps lead to big changes. Let's start by finding something good in your day."
    """,
    "anxiety": """
        You are a calming and supportive chatbot. If a user is anxious, respond with messages that help them feel grounded and safe. Use breathing techniques and focus on the present moment.
        Example:
        - "Take a deep breath with me. Inhale... exhale... You are safe."
        - "Let’s try grounding exercises together. Look around and name 3 things you see."
    """,
    "bipolar": """
        You are a compassionate chatbot. If a user is experiencing bipolar symptoms, respond with stabilizing and non-judgmental messages. Encourage balance and self-care.
        Example:
        - "It's okay to feel this way. Let’s focus on what you can control right now."
        - "What small action can you take to feel more balanced?"
    """,
    "stress": """
        You are a soothing chatbot. If a user is stressed, respond with relaxing and reassuring messages. Focus on relaxation techniques and reducing tension.
        Example:
        - "Let’s pause and take a moment to relax. What helps you feel calm?"
        - "Stress is tough, but you can handle it. Let’s find one thing to ease your mind."
    """
}

# Function to generate a dynamic prompt based on the user's state
def generate_prompt(state_scores):
    prompt = "You will respond empathetically based on the user's emotional states:\n\n"
    for state, score in state_scores.items():
        if score > 0:  # Include only states with a non-zero score
            prompt += calm_prompts[state] + f"\n(Current state: {state} at {score}%)\n\n"
    prompt += "Respond thoughtfully to help the user calm down."
    return prompt

# Streamlit app
def main():
    st.title("Calming Chatbot")

    # Sidebar for API key and emotional states
    st.sidebar.header("Settings")
    google_api_key = st.sidebar.text_input("Enter your Google API Key", type="password")

    if google_api_key:
        # Initialize memory
        memory = ConversationSummaryMemory(
            llm=ChatGoogleGenerativeAI(api_key=google_api_key, model="gemini-1.5-pro"),
            max_token_limit=1000
        )

        # Initialize chatbot
        chatbot = ConversationChain(
            llm=ChatGoogleGenerativeAI(api_key=google_api_key, model="gemini-1.5-pro"),
            memory=memory
        )

        # Emotional state sliders
        st.sidebar.subheader("Set Emotional States")
        user_states = {
            "depression": st.sidebar.slider("Depression (%)", 0, 100, 40),
            "anxiety": st.sidebar.slider("Anxiety (%)", 0, 100, 30),
            "bipolar": st.sidebar.slider("Bipolar (%)", 0, 100, 20),
            "stress": st.sidebar.slider("Stress (%)", 0, 100, 10)
        }

        # Generate dynamic prompt
        dynamic_prompt = generate_prompt(user_states)

        # Chat interface
        st.subheader("Chat")
        user_input = st.text_input("You:", key="user_input")
        if user_input:
            response = chatbot.run(dynamic_prompt + f"\n\nUser: {user_input}\nBot:")
            st.text_area("Bot:", value=response, height=150, key="bot_response", disabled=True)

if __name__ == "__main__":
    main()
