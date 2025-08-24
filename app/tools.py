# backend/app/tools.py
import os
# Updated import: Using the recommended TavilySearch from langchain_tavily
from langchain_tavily import TavilySearch # Corrected import
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_tavily_search_tool():
    """
    Returns a configured Tavily Search tool.
    This tool uses the TAVILY_API_KEY from the environment.
    """
    # Initialize TavilySearch with your API key
    # max_results can be adjusted based on how many search results you want to retrieve
    # The class name has changed from TavilySearchResults to TavilySearch
    return TavilySearch(max_results=5, api_key=os.getenv("TAVILY_API_KEY"))

# You can add other tools here as needed, for example:
# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper

# def get_wikipedia_tool():
#     """
#     Returns a configured Wikipedia search tool.
#     """
#     return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

