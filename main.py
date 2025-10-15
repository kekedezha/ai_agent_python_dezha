from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Load environment variables from .env file, this icludes API keys for LLMs
load_dotenv() 

## STRUCTURED OUTPUT WITH PydanticOutputParser

# Class to specify fields that you want as output from the LLM
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools: list[str]

# Set up LLMs
llm = init_chat_model("claude-3-7-sonnet-20250219", model_provider="anthropic")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

##PROMPT TEMPLATE
prompt_template = ChatPromptTemplate.from_messages(
    [
        # System message to LLM so it knows its role
        ("system",
      """
      You are a research assistant that will help generate a research paper. 
      Answer the user query and use necessary tools. 
      Wrap the output in this format and provide no other text\n{format_instructions}
      """),
      #
      ("placeholder", "{chat_history}"),
      ("human", "{query}"),
      ("placeholder", "{agent_scratchpad}"),
      ]
).partial(format_instructions=parser.get_format_instructions())

## CREATE AGENT
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt_template,
    tools=[],
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": "Give a list of the most important events in AI from 2020 to 2024"})
print(raw_response)