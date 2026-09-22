from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

tavily = TavilyClient()

def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

def main() -> None:
    llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash")
    tools = [search]
    agent = create_agent(model=llm,tools=tools)
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain fully remote, accepting applicants right now on linkedin and list their details")})
    print(result["messages"][-1].text)

if __name__ == "__main__":
    main()
