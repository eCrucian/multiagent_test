import os
import requests
import time
import json

# --- Configuration for Ollama ---
# Ensure your local Ollama server is running.
OLLAMA_URL = "http://localhost:11434/api/generate"
# Specify the model you have downloaded and want to use, e.g., "llama3", "mistral", etc.
OLLAMA_MODEL = "deepseek-r1" 

# --- Agent Definitions ---
# Each agent is a dictionary containing its name, a description of its expertise,
# and a list of keywords to trigger its selection.
AGENTS = [
    {
        'name': 'Process Organizer',
        'description': 'Organizes a process into clear, sequential steps based on a given description.',
        'keywords': ['organize', 'process', 'steps', 'workflow', 'sequence', 'map'],
    },
    {
        'name': 'Process Synthesizer',
        'description': 'Reviews and adjusts a process to eliminate redundancies, making it more streamlined and simple while achieving the same goal.',
        'keywords': ['synthesize', 'streamline', 'simplify', 'optimize', 'process', 'redundant', 'efficiency'],
    },
    {
        'name': 'Risk Identifier',
        'description': 'Identifies operational risks in each step of a process and suggests mitigation strategies.',
        'keywords': ['risk', 'identify', 'operational risk', 'mitigate', 'threat', 'vulnerability', 'assessment'],
    },
    {
        'name': 'Control Designer',
        'description': 'Given the risks in a process and based on classic systems control theory, suggests controls for each evaluated risk.',
        'keywords': ['control', 'review', 'design', 'control theory', 'sensor', 'input', 'output', 'controller', 'implement'],
    },
    {
        'name': 'Control Synthesizer',
        'description': 'Evaluates all controls in a process, suggesting simplifications, mergers, or redesigns to improve efficiency.',
        'keywords': ['synthesize', 'simplify', 'merge', 'evaluate', 'control', 'optimize', 'redesign'],
    },
    {
        'name': 'Control Writer',
        'description': 'Given a control, it ensures that the control is written in a clear, actionable way that specifies who, where, why, what, when, and how (5W1H).',
        'keywords': ['write', 'document', 'control', 'procedure', 'who', 'what', 'where', 'when', 'why', 'how'],
    },
    {
        'name': 'Regulatory Checker',
        'description': 'Given a process with its controls and risks, this agent checks if regulatory compliance requirements are satisfied. If not, it indicates the gaps.',
        'keywords': ['regulatory', 'compliance', 'check', 'audit', 'regulation', 'satisfied', 'requirements', 'gap'],
    },
    {
        'name': 'Regulatory Interpreter',
        'description': 'Given a regulation text, it determines and extracts the specific, actionable requirements that must be met in a process.',
        'keywords': ['interpret', 'regulation', 'requirements', 'legal', 'rule', 'law', 'extract', 'determine'],
    }
]

# --- Core Logic ---

def call_ollama_api(prompt: str, retries: int = 3, delay: int = 5) -> str:
    """
    Calls the local Ollama API with a given prompt.
    Includes error handling and exponential backoff for retries.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False  # We want the full response at once
    }

    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            text_response = result.get('response', "Error: No 'response' key found in Ollama API output.")
            
            return text_response

        except requests.exceptions.RequestException as e:
            print(f"Ollama API call failed on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                print("Ollama API call failed after multiple retries.")
                raise
        except (KeyError, IndexError) as e:
            print(f"Error parsing Ollama API response: {e}")
            print(f"Full response content: {response.text}")
            raise Exception("Failed to parse Ollama API response.")

def select_agents_with_llm(task: str) -> list:
    """
    Uses an LLM to select the most appropriate agents based on the task's context.
    """
    print("\n--- Selecting Initial Agents with LLM ---")
    
    agent_descriptions = "\n".join([f"- {agent['name']}: {agent['description']}" for agent in AGENTS])
    
    prompt = f"""You are an expert dispatcher. Based on the user's task, select a team of specialized AI agents.
Here are the available agents:
{agent_descriptions}

The user's task is: "{task}"

Which agent(s) are the most appropriate for this team? 
Your answer MUST be a JSON-formatted list of agent names.
Example: ["Creative Writer", "Code Generator"]
If no agent is suitable, return an empty list []."""

    try:
        response_text = call_ollama_api(prompt)
        cleaned_response = response_text.strip().replace("```json", "").replace("```", "")
        selected_agent_names = json.loads(cleaned_response)
        
        selected_agents = {agent['name']: agent for agent in AGENTS if agent['name'] in selected_agent_names}
        return selected_agents
    except json.JSONDecodeError:
        print("Error: LLM did not return valid JSON for agent selection. Falling back to keyword search.")
        lower_case_task = task.lower()
        keyword_selected = [agent for agent in AGENTS if any(keyword in lower_case_task for keyword in agent['keywords'])]
        return {agent['name']: agent for agent in keyword_selected}
    except Exception as e:
        print(f"An error occurred during agent selection: {e}")
        return {}

def planner_agent(task: str, conversation_history: str, available_agents: dict) -> dict:
    """
    The planner/orchestrator agent that decides the next step.
    """
    print("\n--- Planner Agent is thinking... ---")
    agent_names = list(available_agents.keys())
    
    prompt = f"""You are the planner/orchestrator for a team of AI agents. Your job is to determine the next step to solve the user's task.

User's Task: "{task}"

Conversation History:
{conversation_history}

Available Agents: {agent_names}

Based on the conversation, what is the very next action to take?
If the task is complete and ready for a final answer, respond with an agent named "Synthesizer".
Otherwise, choose the best agent from the available list to continue the work.

Your response must be a single JSON object with two keys: "next_agent" and "sub_task".
The "sub_task" should be a clear, concise instruction for the chosen agent.

Example:
{{
  "next_agent": "Code Generator",
  "sub_task": "Generate a Python function that creates a simple melody based on the story's theme of discovery."
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

def run_multi_agent_system(task: str):
    """
    Orchestrates the multi-agent process from task input to final answer.
    """
    print("="*50)
    print(f"🎬 Starting Task: \"{task}\"")
    print("="*50)

    available_agents = select_agents_with_llm(task)
    if not available_agents:
        print("\n🤔 No suitable agents found for this task.")
        # Simplified suggestion for Ollama version
        print("\nI'm not able to do that. May I suggest rephrasing your request or asking for a task related to research, creative writing, coding, or planning?")
        return

    print(f"\n👥 Initial Agent Team: {list(available_agents.keys())}")
    
    collaboration_history = f"The user's original task is: \"{task}\"\n\n"
    
    max_turns = 5 # Safety break to prevent infinite loops
    for turn in range(max_turns):
        decision = planner_agent(task, collaboration_history, available_agents)
        next_agent_name = decision.get("next_agent")
        sub_task = decision.get("sub_task", "No sub-task provided.")

        print(f" Planner decided on: {next_agent_name} | Task: {sub_task}")

        if next_agent_name == "Synthesizer":
            print("\n--- Synthesizing Final Answer ---")
            break # Exit loop to start synthesis
        
        agent = available_agents.get(next_agent_name)
        if not agent:
            print(f"Warning: Planner chose an unavailable agent '{next_agent_name}'. Ending.")
            break
            
        print(f"\n--- Consulting {agent['name']} ---")
        prompt = f"""You are '{agent['name']}', with expertise in: "{agent['description']}".
The overall user task is: "{task}"

Conversation History:
{collaboration_history}

Your specific sub-task now is: "{sub_task}"
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
    final_prompt = f"""You are a master synthesizer. Combine the contributions from the conversation into a single, cohesive, and comprehensive final answer for the user.
Do not just list the contributions. Weave them together into a final, well-structured response that directly addresses the user's original request.

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
    # Example task to run the system
    example_task = "Write a short story about a robot who discovers music, and include a python code snippet for generating a simple melody."
    
    # You can get the task from user input like this:
    # user_task = input("Please enter your task: ")
    # run_multi_agent_system(user_task)
    
    run_multi_agent_system(example_task)


