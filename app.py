from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Include citations to support the results",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,stock_fundamentals=True,company_info=True)],
    instructions="Include citations to support the results",
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Include citations to support the results", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

companies = ["Reliance", "Airtel", "Vodafone"]

for company in companies:
    query = (
        f"Provide area-wise performance of telecom {company} by analyzing recent trends "
        "thoroughly to help the user choose the right network. "
        "Also provide financial data to help B2B."
    )
    agent_team.print_response(query, stream=True)