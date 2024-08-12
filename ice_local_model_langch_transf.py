from dotenv import load_dotenv
import torch
import transformers
from transformers import AutoTokenizer
from transformers import pipeline
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate

if __name__ == "__main__":
    load_dotenv()

    print("Hello World")

    model = "C:/Users/I038077/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            max_length=3000,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # We add a template to the prompt
    # This template serves as a blueprint to direct the summarization process and ensure the encapsulation of key pointsss.
    template = """
              Write a summary of the following text delimited by triple backticks.
              Return your response which covers the key points of the text.
              ```{text}```
              SUMMARY:
           """
    
    pt = PromptTemplate(template=template, input_variables=["text"])

    text = """
        Elon Reeve Musk (/ˈiːlɒn/; EE-lon; born June 28, 1971) is a businessman and investor. He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect and former chairman of Tesla, Inc.; owner, chairman and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. He is the wealthiest person in the world, with an estimated net worth of US$232 billion as of December 2023, according to the Bloomberg Billionaires Index, and $254 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.[5][6]
        A member of the wealthy South African Musk family, Elon was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania, and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University. However, Musk dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999, and, that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.
        In October 2002, eBay acquired PayPal for $1.5 billion, and that same year, with $100 million of the money he made, Musk founded SpaceX, a spaceflight services company. In 2004, he became an early investor in electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.). He became its chairman and product architect, assuming the position of CEO in 2008. In 2006, Musk helped create SolarCity, a solar-energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year, Musk co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and the Boring Company, a tunnel construction company. In 2022, he acquired Twitter for $44 billion. He subsequently merged the company into newly created X Corp. and rebranded the service as X the following year. In March 2023, he founded xAI, an artificial intelligence company.
    """

    # LLMChain consists of a prompt template and a language model pipeline
    # LLMChain provides a wrapper on top of a model to generate responses based on a given prompt. 
    # LLMChain wrapper could work with multiple language models and pipelines
    # Various possibilities could be to use Transformers pipeline, HuggingFacePipeline or the OpenAIGPTPipeline

    # Somehow Langchain does not support Transformers pipeline. Hence, we directly used Transformers pipeline without Langchain in "ice_local_model_transf_only.py"
    # Because of which following does not work
    # llm_chain = LLMChain(prompt=prompt, llm=pipeline)

    # Hence, we will make use HuggingFacePipeline instead of Transformers pipeline

    # Please check following links on how to work with Langchain
    # https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/
    # https://api.python.langchain.com/en/latest/llms/langchain_huggingface.llms.huggingface_pipeline.HuggingFacePipeline.html

    hfgppl = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})
    llm_chain = LLMChain(prompt=pt, llm=hfgppl)

    print(llm_chain.run(text))

    # Other possibility is to directly work with HuggingFacePipeline + Langchain (without Tranformers). You can check this in ice_local_model_langch_only.py
    # This would be the best way to work with Langchain and HuggingFacePipeline  