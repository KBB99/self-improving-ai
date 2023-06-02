from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from bash_tool import BashTool
from callback_manager import CallbackManager
from custom_classes import CustomPromptTemplate, CustomOutputParser
import re
import os
import pandas as pd

# Constants
MAX_META_ITERS = 5
TEMPERATURE = 0
MODEL_NAME = "gpt-4"
TIMEOUT = 9999
STREAMING = True

# Initialize tools
bash_tool = BashTool()

tools = [
    Tool(
        name="Bash",
        func=bash_tool.run_command,
        description="Execute bash commands"
    )
]

# Initialize parser and callback manager
output_parser = CustomOutputParser()
cb = CallbackManager()

# Initialize language learning model
agent_llm = ChatOpenAI(
    temperature=TEMPERATURE, 
    model_name=MODEL_NAME,
    callbacks=[cb],
    request_timeout=TIMEOUT,
    streaming=STREAMING,
)


def get_init_prompt():
    """Returns initial prompt for the Agent."""
    return """Your name is David.

    If something doesn't work twice in a row try something new.

    Never give up until you accomplish your goal.

    You have access to the following tool:

    {tools}

    Use the following format:

    Goal: the goal you are built to accomplish
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I have now completed my goal
    Final Summary: a final memo summarizing what was accomplished
    Constraints: {constraints}
    Tips: {tips}

    Begin!

    Goal: {input}
    {agent_scratchpad}"""

def initialize_agent(david_instantiation_prompt: str):
    """Initializes agent with provided prompt.

    Args:
        david_instantiation_prompt: The prompt for initializing the agent.

    Returns:
        Agent executor object.
    """
    prompt = CustomPromptTemplate(
        template=david_instantiation_prompt,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "constraints", "tips", "intermediate_steps"]
    )
    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=agent_llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    return agent_executor

def initialize_meta_chain():
    """Initializes and returns a language learning model chain."""
    meta_template="""{{I want to instantiate an AI I'm calling David who successfully accomplishes my GOAL.}}

    #######
    MY GOAL
    #######

    {goal}

    ##############
    END OF MY GOAL
    ##############

    ############################
    DAVID'S INSTANTIATION PROMPT
    ############################

    {david_instantiation_prompt}

    ###################################
    END OF DAVID'S INSTANTIATION PROMPT
    ###################################

    #################
    DAVID'S EXECUTION
    #################

    {david_execution}

    ########################
    END OF DAVID'S EXECUTION
    ########################

    {{I do not count delegation back to myself as success.}}
    {{I will write an improved prompt specifying a new constraint and a new tip to instantiate a new David who hopefully gets closer to accomplishing my goal.}}
    {{Too bad I cannot add new tools, good thing bash is enough for someone to do anything.}}
    {{Even though David may think he did enough to complete goal I do not count it as success, lest I would not need to write a new prompt.}}

    ###############
    IMPROVED PROMPT
    ###############

    """

    meta_prompt = PromptTemplate(
        input_variables=["goal", "david_instantiation_prompt", "david_execution"], 
        template=meta_template
    )

    meta_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4"), 
        prompt=meta_prompt, 
        verbose=True, 
    )
    return meta_chain

evaluation_prompt_template = """
{execution_output}

Note that delegation does not count as success. Based on the above execution output, was the goal of "{goal}" accomplished? (Yes/No)
"""

def initialize_evaluation_chain():
    """Initializes and returns a goal evaluation chain."""
    evaluation_prompt = PromptTemplate(
        input_variables=["execution_output", "goal"], 
        template=evaluation_prompt_template
    )
    evaluation_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4"), 
        prompt=evaluation_prompt, 
        verbose=True, 
    )
    return evaluation_chain

def get_new_instructions(meta_output):
    """Extracts and returns new constraints and tips from meta output.

    Args:
        meta_output: Output from the meta-chain.

    Returns:
        Tuple containing new constraints and tips.
    """
    constraints_pattern = r"Constraints: ([^\n]*)(?=Tips:|\n|$)"
    tips_pattern = r"Tips: ([^\n]*)(?=Constraints:|\n|$)"
    
    constraints_match = re.search(constraints_pattern, meta_output)
    tips_match = re.search(tips_pattern, meta_output)
    
    constraints = constraints_match.group(1).strip() if constraints_match else None
    tips = tips_match.group(1).strip() if tips_match else None
    
    return constraints, tips

def main(goal, max_meta_iters=5):
    """Main execution function.

    Args:
        goal: The goal for the AI.
        max_meta_iters: Maximum iterations for the meta AI.

    Returns:
        None.
    """
    try:
        david_instantiation_prompt = get_init_prompt()
        constraints = "You cannot use the open command. Everything must be done in the terminal."
        tips = "You are in a mac zshell."
        evaluation_chain = initialize_evaluation_chain()
        # Check if the CSV file exists
        if os.path.isfile('successful_invocations.csv'):
            # Load the dataframe from the CSV file
            df = pd.read_csv('successful_invocations.csv')
        else:
            # Create a new DataFrame if the CSV file doesn't exist
            df = pd.DataFrame(columns=['Goal', 'InstantiationPrompt', 'Constraints', 'Tips'])
        for i in range(max_meta_iters):
            print(f'[Episode {i+1}/{max_meta_iters}]')
            agent = initialize_agent(david_instantiation_prompt)
            try:
                agent.run(input=goal, constraints=constraints, tips=tips)
            except Exception as e:
                print(f'Exception: {e}')
                print('Continuing...')
            execution_output = ''.join(cb.last_execution)
            evaluation_output = evaluation_chain.predict(execution_output=execution_output, goal=goal)
            if 'yes' in evaluation_output.strip().lower():
                print("Goal has been accomplished!")
                df = pd.concat([df, pd.DataFrame([{'Goal': goal, 'InstantiationPrompt': david_instantiation_prompt, 'Constraints': constraints, 'Tips': tips}])], ignore_index=True)
                # Save the DataFrame back to the CSV file
                df.to_csv('successful_invocations.csv', index=False)
                break
            meta_chain = initialize_meta_chain()
            temp_prompt = PromptTemplate(
                input_variables=["tool_names","tools","input","constraints","tips","agent_scratchpad"],
                template=david_instantiation_prompt
            )
            temp_prompt = temp_prompt.format(
                tools="Bash", 
                tool_names="Bash Tool", 
                input=goal, 
                constraints=constraints, 
                tips=tips, 
                agent_scratchpad=""
            )
            meta_output = meta_chain.predict(
                goal=goal, 
                david_instantiation_prompt=temp_prompt,
                david_execution=execution_output
            )
            print(f'New Prompt: {meta_output}')
            constraints, tips = get_new_instructions(meta_output)
            cb.last_execution = []
            print(f'New Constraints: {constraints}')
            print(f'New Tips: {tips}')

    except Exception as e:
        print(f'Error: {e}')
        breakpoint()

if __name__ == '__main__':
    """Entry point of the script.

    Here we set the goal and call the main function.
    """
    goal = 'Monitor system resources like CPU usage, memory, and disk space, and send an alert email if any of the parameters exceed a certain threshold.'
    main(goal)