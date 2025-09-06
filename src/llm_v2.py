import os
import requests
import time
import json

# --- Required Installations ---
# This script requires additional libraries to read files.
# Please install them using pip:
# pip install pypdf python-docx

try:
    import pypdf
    import docx
except ImportError:
    print("="*50)
    print("IMPORTANT: Missing required libraries.")
    print("Please run the following command to install them:")
    print("pip install pypdf python-docx")
    print("="*50)
    exit()

# --- Configuration for Ollama ---
# Ensure your local Ollama server is running.
OLLAMA_URL = "http://localhost:11434/api/generate"
# Specify the model you have downloaded and want to use, e.g., "llama3", "mistral", etc.
OLLAMA_MODEL = "llama3" 

# --- Agent Definitions ---
# Each agent is a dictionary containing its name and a description of its expertise.
AGENTS = [
    {
        'name': 'Process Organizer',
        'description': 'Organizes a process into clear, sequential steps based on a given description.',
    },
    {
        'name': 'Process Synthesizer',
        'description': 'Reviews and adjusts a process to eliminate redundancies, making it more streamlined and simple while achieving the same goal.',
    },
    {
        'name': 'Risk Identifier',
        'description': 'Identifies operational risks in each step of a process and suggests mitigation strategies.',
    },
    {
        'name': 'Control Designer',
        'description': 'Given the risks in a process and based on classic systems control theory, suggests controls for each evaluated risk.',
    },
    {
        'name': 'Control Synthesizer',
        'description': 'Evaluates all controls in a process, suggesting simplifications, mergers, or redesigns to improve efficiency.',
    },
    {
        'name': 'Control Writer',
        'description': 'Given a control, it ensures that the control is written in a clear, actionable way that specifies who, where, why, what, when, and how (5W1H).',
    },
    {
        'name': 'Regulatory Checker',
        'description': 'Given a process with its controls and risks, this agent checks if regulatory compliance requirements are satisfied. If not, it indicates the gaps.',
    },
    {
        'name': 'Regulatory Interpreter',
        'description': 'Given a regulation text, it determines and extracts the specific, actionable requirements that must be met in a process.',
    }
]

# --- Core Logic ---

def read_file_content(filepath: str) -> str:
    """Reads content from various file types and returns it as a string."""
    _, extension = os.path.splitext(filepath.lower())
    content = ""
    try:
        if extension == '.txt' or extension == '.md':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        elif extension == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content = json.dumps(data, indent=2) # Pretty print for LLM readability
        elif extension == '.docx':
            doc = docx.Document(filepath)
            content = "\n".join([para.text for para in doc.paragraphs])
        elif extension == '.pdf':
            with open(filepath, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    content += page.extract_text() or ""
        else:
            print(f"Warning: Unsupported file type '{extension}'. File will be ignored.")
            return ""
        print(f"\n--- Successfully read content from {filepath} ---")
        return content
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return ""
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        return ""

def call_ollama_api(prompt: str, retries: int = 3, delay: int = 5) -> str:
    """
    Calls the local Ollama API with a given prompt.
    Includes error handling and exponential backoff for retries.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_URL, headers=headers, json=payload, timeout=180) # Increased timeout
            response.raise_for_status()
            result = response.json()
            return result.get('response', "Error: No 'response' key found.")
        except requests.exceptions.RequestException as e:
            print(f"Ollama API call failed on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                print("Ollama API call failed after multiple retries.")
                raise

def select_agents_with_llm(task: str, context: str) -> dict:
    """
    Uses an LLM to select the most appropriate agents based on the task and context.
    """
    print("\n--- Selecting Initial Agents with LLM ---")
    agent_descriptions = "\n".join([f"- {agent['name']}: {agent['description']}" for agent in AGENTS])
    
    prompt = f"""You are an expert dispatcher. Based on the user's task and provided context, select a team of specialized AI agents.
Here are the available agents:
{agent_descriptions}

The user's task is: "{task}"
Context from provided documents: "{context[:1000]}..." # Truncated for prompt brevity

Which agent(s) are the most appropriate for this team? 
Your answer MUST be a JSON-formatted list of agent names. Do not add explanation.
Example: ["Regulatory Interpreter", "Risk Identifier"]
If no agent is suitable, return an empty list []."""

    try:
        response_text = call_ollama_api(prompt)
        cleaned_response = response_text.strip().replace("```json", "").replace("```", "")
        selected_agent_names = json.loads(cleaned_response)
        selected_agents = {agent['name']: agent for agent in AGENTS if agent['name'] in selected_agent_names}
        return selected_agents
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error during LLM agent selection: {e}. Cannot proceed.")
        return {}

def planner_agent(task: str, conversation_history: str, available_agents: dict) -> dict:
    """
    The planner/orchestrator agent that decides the next step.
    """
    print("\n--- Planner Agent is thinking... ---")
    agent_names = list(available_agents.keys())
    
    prompt = f"""You are the planner for a team of AI agents. Your job is to determine the next step to solve the user's task.

User's Task: "{task}"

Conversation History (includes document context):
{conversation_history}

Available Agents: {agent_names}

Based on the history, what is the very next action?
If the task is complete, respond with an agent named "Synthesizer".
Otherwise, choose the best agent from the available list to continue.

Your response must be a single JSON object with "next_agent" and "sub_task" keys.
The "sub_task" should be a clear, concise instruction for the chosen agent.

Example:
{{
  "next_agent": "Risk Identifier",
  "sub_task": "Based on the interpreted regulatory requirements, identify operational risks in the user onboarding process."
}}
"""
    try:
        response_text = call_ollama_api(prompt)
        cleaned_response = response_text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(cleaned_response)
        return decision
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error in planner agent decision: {e}. Defaulting to Synthesizer.")
        return {"next_agent": "Synthesizer", "sub_task": "Synthesize the final answer from the available information."}

def run_multi_agent_system(task: str, filepath: str = None):
    """
    Orchestrates the multi-agent process from task input to final answer.
    """
    print("="*50)
    print(f"🎬 Starting Task: \"{task}\"")
    if filepath:
        print(f"📄 Using File: \"{filepath}\"")
    print("="*50)
    
    file_context = ""
    if filepath:
        file_context = read_file_content(filepath)
        if not file_context:
            print("Could not read file. Proceeding without file context.")

    available_agents = select_agents_with_llm(task, file_context)
    if not available_agents:
        print("\n🤔 No suitable agents found for this task.")
        print("\nPlease rephrase your request or describe a task related to process organization, risk, controls, or regulatory compliance.")
        return

    print(f"\n👥 Initial Agent Team: {list(available_agents.keys())}")
    
    collaboration_history = f"The user's task is: \"{task}\"\n\n"
    if file_context:
        collaboration_history += f"The user provided this document:\n--- DOCUMENT START ---\n{file_context}\n--- DOCUMENT END ---\n\n"
    
    max_turns = 7 # Increased turns for more complex tasks
    for turn in range(max_turns):
        decision = planner_agent(task, collaboration_history, available_agents)
        next_agent_name = decision.get("next_agent")
        sub_task = decision.get("sub_task", "No sub-task provided.")

        print(f" Planner decided on: {next_agent_name} | Task: {sub_task}")

        if next_agent_name == "Synthesizer":
            print("\n--- Task complete. Synthesizing Final Answer ---")
            break
        
        agent = available_agents.get(next_agent_name)
        if not agent:
            print(f"Warning: Planner chose an unavailable agent '{next_agent_name}'. Ending.")
            break
            
        print(f"\n--- Consulting {agent['name']} ---")
        prompt = f"""You are '{agent['name']}', with expertise in: "{agent['description']}".
The overall user task is: "{task}"

Conversation History (includes full document context):
{collaboration_history}

Your specific sub-task is: "{sub_task}"
Provide a focused contribution based ONLY on your sub-task.
---
Your Contribution:"""
        
        try:
            contribution = call_ollama_api(prompt)
            print("💬 Contribution received.")
            collaboration_history += f"Contribution from {agent['name']} (Task: {sub_task}):\n{contribution}\n\n"
        except Exception as e:
            print(f"Error getting contribution from {agent['name']}: {e}")
            collaboration_history += f"Note: The {agent['name']} failed to contribute due to an error.\n\n"

    # Final Synthesis
    final_prompt = f"""You are a master synthesizer. Combine the contributions from the conversation into a single, cohesive final answer for the user.
Weave the information together into a well-structured response that directly addresses the user's original request.

Original Task: "{task}"

Agent Conversation History:
{collaboration_history}
---
Final Synthesized Response:"""

    try:
        final_answer = call_ollama_api(final_prompt)
        print("\n" + "="*50)
        print("✅ FINAL RESULT")
        print("="*50)
        print(final_answer)
    except Exception as e:
        print(f"\n❌ Error generating final answer: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Multi-Agent System with local Ollama model.")
    print(f"Make sure your Ollama server is running and the model '{OLLAMA_MODEL}' is available.")
    
    # Create a dummy regulation file for the example
    dummy_regulation_text = """
    Regulation XYZ-2025: Data Privacy and User Onboarding

    Article 1: All user data must be encrypted at rest.
    Article 2: During onboarding, users must be explicitly asked for consent before collecting personal information.
    Article 3: A clear and accessible privacy policy must be available to the user.
    Article 4: The user has the right to request data deletion at any time.
    """
    dummy_filepath = "regulation_example.txt"
    with open(dummy_filepath, "w") as f:
        f.write(dummy_regulation_text)

    # Example task that uses the file
    example_task = f"Given the regulation provided in the file, first interpret the requirements, and then design appropriate controls for a new user onboarding process."
    example_filepath = dummy_filepath
    
    run_multi_agent_system(example_task, filepath=example_filepath)

