import os
from dotenv import load_dotenv
from langchain import LLMChain
from langchain import HuggingFacePipeline, PromptTemplate
import transformers
from transformers import AutoTokenizer
import torch
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup

if __name__ == "__main__":
    # load_dotenv()

    summary_template = """
        given the information {information} about a person I want you to create:
        1. A short summary
        2. two interesting facts about them
    """

    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)

    # We have created a pipeline using tranformers  
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=30000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    # We could cutomise the output using the model_kwargs
    # A temperature of 0 gives direct summaries, while higher values add some randomness.
    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm_chain = LLMChain(prompt=summary_prompt_template, llm=llm)

    # We initally started with the hardcoded Linkedin profile URL though the data corresponding to the profile is still hard coded data in GIST fpr Edne Marco
    # information = scrape_linkedin_profile(linkedin_url="https://www.linkedin.com/in/elonmusk/")
    
    # Now instead of hard coding the URL we want to get the URL against the name of the person
    # For this, we will make use of Agentssss
    linkednin_profile_url = lookup(name="Eden Marco")

    # Now we will scrape the information from the linkedin profile
    information = scrape_linkedin_profile(linkedin_url=linkednin_profile_url)

    print(information)

    # We will now pass the information to the LLMChain to get the summary
    res = llm_chain.invoke({"information": information})

    print(res)
