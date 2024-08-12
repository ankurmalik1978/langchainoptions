from dotenv import load_dotenv

load_dotenv()

# LLM modelsdont have most uptodate information for e.g. "how is the weather today?" 
# We could use Agents which could help in performing such tasks like getting the weather information, call APIs, accessing the web, accessing database, etc.
# Agents are a way to encapsulate a set of tools and a language model to perform a specific task.
# Agents could interact with LLM. They could perform a task and return the result to LLM. All this could be chained together to perform a complex task
# Agents are also internally using LLM to perform the task. When an agent is invoked, it uses the language model to identigy sub-tasks and perform the task.
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_core.tools import Tool
#from langchain_openai import ChatOpenAI
from langchain import hub
from langchain import HuggingFacePipeline, PromptTemplate
from tools.tools import get_profile_url
import torch
import transformers
from transformers import AutoTokenizer

def lookup(name: str) -> str:

    model = "C:/Users/I038077/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"

    # model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)

    # # # We have created a pipeline using tranformers  
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10000
        # torch_dtype=torch.bfloat16,
        # trust_remote_code=True,
        # device_map="auto",
        # max_length=30000,
        # do_sample=True,
        # top_k=10,
        # num_return_sequences=1,
        # eos_token_id=tokenizer.eos_token_id
    )

    # We could cutomise the output using the model_kwargs
    # A temperature of 0 gives direct summaries, while higher values add some randomness.
    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0, 'is_split_into_words':True})

    # llm = HuggingFacePipeline.from_model_id(
    #     model_id=model,
    #     task="text-generation",
    #     model_kwargs = {'temperature':0}
    # )

    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                          Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    # Agents are a way to encapsulate a set of tools and a language model to perform a specific task.
    # Agents will have in their toolbox a set of tools that could be used to perform the tasks
    # Tools are the place where the actual logic is implemented for interacting with the external world
    # Tools contains set of functions that could be used to perform the task. These functions have a description which is then used to perform the task
    # Power of langchain is that we could convert any function in Python into a Langchain Toola and make it accessible to LLM 
    # In this case, we have a tool that could be used to get the Linkedin profile URL
    # Agent will receive a name and it will be able to output linkedin profile URL  

    # create_react_agent is an inbuilt function in Langchain 
    # It takes as input llm, tools and prompt and creates an agent which could be used to perform the task 
    # ReAct is a framewrok where LLMs are used to generate both reasoning traces and task-specific actions in an interleaved manner

    # In our case, we want to determoine the linkedin profile URL of a person given their name
    # We will make use of ReactAgent which has a Search Tool in its Toolbox 
    # Agent will reason and calculate whats steps it need to perform to perform the task
    # It will know that it has seraching capability in its Toolbox and it will use that to perform the task

    # Tool is the interface between the agent and the external world for e.g. Search online, call APIs, access database, etc.
    # It has a function to execute for e..g. search online, call APIs, access database, etc. 
    # It has a description which describes what function does and what it the Output of the function

    # There are 3 attributes passed to the Tool
    # name: name of the tool, this appears in the logs and is used to identify the tool. Hence, it should be unique and meaningful
    # func: function to be executed when the tool is invoked
    # description: This is very important since on the basis of the description LLM will decide whether to use this Tool or not. It should have as much information as possible
    # If the LLM decides to run this tool on the basis of the description, it will execute the function 
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    # We use Hub to download pre-built Prompts by the Community 
    # This the prompt used by the Agent to perform the task and is the reasoning engine for the Agent. Its based on COT
    # This send to LLM and it includes Tool
    # This is one of the many prompts available. So, we could use any custom prompt as well
    react_prompt = hub.pull("hwchase17/react")

    # React is the most powerfull way for implementing agents in Langchanin
    # create_react_agent is a function which creates an agent using React framework
    # This takes as input
    # llm: The language model which will be used by the agent
    # tools: The tools which are available to the agent
    # prompt: The prompt which will be used by the agent to perform the task
    # Output of this function is an Agent which could be used to perform the task
    # Using LLM and Tools, Agent will determine the steps and the Tool to be used to perform the task
    # There might be many Tools in the Toolbox of the Agent but it will decide which Tool to use based on the reasoning --> "React"
    # For the reasoning, it will use the Prompt, LLM and the description of the Tool 
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    # runtime of the Agent
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, handle_parsing_errors=True, verbose=True)

    # This is the place where the Agent is invoked to perform the task
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url