# Nothing much here we need agentic search engine in order to get the data in the formate which the 
# Agent requires

# libraries
from dotenv import load_dotenv
import os
from langchain_tavily import TavilySearch
from tavily import TavilyClient

# load environment variables from .env file
_ = load_dotenv()

# connect
client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# run search
# result = client.search("who attacked the donald trump while he was giving speech?",
#                        include_answer=True)

# # print the answer
# print(result["answer"])


# choose location (try to change to your own city!)

city = "Chennai"

query = f"""
    what is the current weather in {city}?
    Should I travel there today?
    "weather.com"
"""
# run search
result = client.search(query, max_results=1)

# print first result this gives the regular input from the search result.
data = result["results"][0]["content"]

# print(data)

# We will divide that booring sort of data into beautifully looking JSON data 

import json
from pygments import highlight, lexers, formatters

# parse JSON
parsed_json = json.loads(data.replace("'", '"'))

# pretty print JSON with syntax highlighting
formatted_json = json.dumps(parsed_json, indent=4)
colorful_json = highlight(formatted_json,
                          lexers.JsonLexer(),
                          formatters.TerminalFormatter())

print(colorful_json)    # This gives a beautifully looking JSON output 
