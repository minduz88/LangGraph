import os
from datetime import datetime
from typing import Annotated, TypedDict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import io
import base64
import matplotlib.pyplot as plt
import sqlite3

# LangChain imports
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool

# LangGraph imports
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# OpenAI imports
from openai import OpenAI

# Load environment variables
load_dotenv()
os.environ['LANCHAIN_API_KEY'] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tool definitions
class GenImageSchema(BaseModel):
    prompt: str = Field(description="The prompt for image generation")

@tool(args_schema=GenImageSchema)
def generate_image(prompt: str) -> str:
    """Generate an image using DALL-E based on the given prompt."""
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    return f"Successfully generated the image!,{response.data[0].url}"

def query_apartment_database(query: str) -> str:
    """Execute SQL queries on the apartment sales database"""
    try:
        conn = sqlite3.connect('dataset/database.db')
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        formatted_results = []
        for row in results:
            formatted_row = dict(zip(columns, row))
            formatted_results.append(formatted_row)
        conn.close()
        return str(formatted_results) if formatted_results else "No results found."
    except Exception as e:
        return f"Query execution error: {str(e)}"

# Initialize Python REPL
repl = PythonREPL()

@tool
def python_repl(code: str):
    """Execute Python code."""
    return repl.run(code)

@tool
def data_visualization(code: str):
    """Execute Python code for visualization using matplotlib."""
    try:
        repl.run(code)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        return f"Error creating chart: {str(e)}"

@tool
def search(query: str) -> str:
    """Search the web for information."""
    search_client = TavilySearchResults(max_results=2)
    return search_client.invoke(query)

# System prompt
SYSTEM_PROMPT = f"""
Today is {datetime.now().strftime("%Y-%m-%d")}
You are a helpful AI Assistant that can use various tools:
- Web search tool (Tavily AI API)
- Image generation tool (DALL-E API)
- Code execution tool (Python REPL)
- Data visualization tool (Matplotlib)
- Apartment database query tool (SQL)

You should always answer in the same language as the user's question.
When you can't answer directly, use the appropriate tool:
- For real-time info: use web search
- For visualization: use image generation or data visualization
- For data analysis: use Python REPL
- For apartment information: use apartment database query
"""

def create_agent(docs_info=None, retriever_tool=None):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    tools = [
        generate_image,
        search,
        python_repl,
        data_visualization,
        Tool(
            name="apartment_database_query",
            description="SQL query tool for apartment sales information.",
            func=query_apartment_database
        )
    ]

    if retriever_tool:
        tools.append(retriever_tool)

    llm_with_tools = llm.bind_tools(tools)
    chain = prompt | llm_with_tools

    graph_builder = StateGraph(MessagesState)
    
    def chatbot(state: MessagesState):
        response = chain.invoke(state["messages"])
        return {"messages": response}

    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    return graph_builder.compile()

# Initialize the agent
graph = create_agent()