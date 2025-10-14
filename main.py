from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()  # Load environment variables from .env file

# Set up LLMs
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm2 = ChatAnthropic(model="claude-4-sonnet-latest")


response = llm2.invoke("Hello, how are you? Can you tell me a joke?")
print(response)