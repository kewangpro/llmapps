from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import List, Annotated

# Search tool
search_tool = DuckDuckGoSearchRun(name="Search")

# Web scraping tool
@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        return "\n\n".join([
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ])
    except Exception as e:
        return f"Error scraping webpages: {str(e)}"

# Code execution tool
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
        result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
        return result_str
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

# Combined tool list for easy access
all_tools = [search_tool, scrape_webpages, python_repl_tool]
search_tools = [search_tool]
scraping_tools = [scrape_webpages]
code_tools = [python_repl_tool]