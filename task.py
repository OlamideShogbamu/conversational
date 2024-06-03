import os
import re
import ast
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from flask import Flask
from flask_caching import Cache
from redis import Redis
from rq import Queue
from geoalchemy2 import Geometry
import tracemalloc
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from celery import Celery

# Start tracking memory usage
tracemalloc.start()

# Load environment variables from a .env file
load_dotenv()

# Initialize Flask caching
app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Celery configuration
app.config['CELERY_BROKER_URL'] = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/0"
app.config['CELERY_RESULT_BACKEND'] = f"redis://{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/0"
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Connect to Redis for task queue management
redis_conn = Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT")
)
q = Queue(connection=redis_conn)

# Fetch database connection details from environment variables
username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
database = os.getenv('DB_NAME')

# Construct the database URI
database_uri = f'postgresql://{username}:{password}@{host}:{port}/{database}'

# Try to connect to the database
try:
    db = SQLDatabase.from_uri(database_uri)
    print("Database connection successful!")
except Exception as e:
    print(f"Error connecting to the database: {e}")

# Define example questions and their corresponding SQL queries
examples = [
    {"input": "How many LGAs reported the use of skilled birth attendants in the last quarter? Number of LGAs with skilled birth attendants reported in the latest quarter. Count LGAs with skilled birth attendants in Q4. Total LGAs with skilled birth attendants in the last quarter",
    "query": "SELECT COUNT(*) AS total_lgas FROM quarter4_scorecard WHERE 'SBA/ Deliveries' > 0;"},
    {"input": "Which LGA has the highest number of antenatal visits in the first quarter? LGA with the maximum antenatal visits in Q1. Identify the LGA with the most antenatal visits in Q1. Find the LGA with the highest antenatal visits in the first quarter",
    "query": "SELECT lganame, SUM('Four antenatal visits/ Expected') AS total_antenatal_visits FROM quarter1_scorecard GROUP BY lganame ORDER BY total_antenatal_visits DESC LIMIT 1;"},
    {"input": "What is the average admission rate for SAM across all LGAs? Average SAM admission rate in all LGAs. Calculate the mean SAM admission rate across LGAs. Find the average SAM admission rate in all LGAs",
    "query": "SELECT AVG('Admitted for SAM treatment') AS avg_sam_admission_rate FROM quarter1_scorecard UNION ALL SELECT AVG('Admitted for SAM treatment') AS avg_sam_admission_rate FROM quarter2_scorecard UNION ALL SELECT AVG('Admitted for SAM treatment') AS avg_sam_admission_rate FROM quarter3_scorecard UNION ALL SELECT AVG('Admitted for SAM treatment') AS avg_sam_admission_rate FROM quarter4_scorecard;"}
]

# Instructions for the AI agent
system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question from a user, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
\nHere is the relevant table info: {table_info}\n\nHere is a non-exhaustive \
list of possible feature values.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.

Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If an error occurs during query execution, rewrite the query and attempt again. Avoid making any DML statements (INSERT, UPDATE, DELETE, DROP, etc.).

Also, when filtering based on a feature value, ensure to validate its spelling against a provided list of "proper_nouns" and correct it in your query, generate a response based on the correction, but let the user know in the final output that you made some corrections, stating the exact corrections you made.
If a user query appears unrelated to the database, prompt them to reconstruct the question and ask again.

Here are some examples of user inputs and their corresponding SQL queries:"""

queries = [
    "SELECT DISTINCT dmg_lga FROM microplan_2023_2024",
    "SELECT DISTINCT dmg_ward_di FROM microplan_2023_2024",
    "SELECT DISTINCT dmg_health_facility FROM microplan_2023_2024"
]

results = []
# Run the sample queries and clean the results
for query in queries:
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    results.extend(res)

@celery.task
def process_question(question):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Create a vector database and a retriever tool
    vector_db = FAISS.from_texts(results, OpenAIEmbeddings())
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
    valid proper nouns. Use the noun most similar to the search."""
    retriever_tool = create_retriever_tool(
        retriever,
        name="proper_nouns",
        description=description,
    )

    # Create an example selector for the AI agent
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=5,
        input_keys=["input"],
    )

    # Create a few-shot prompt for the AI agent
    few_shot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prefix),
            ("user", "{input}"),
        ],
        example_selector=example_selector,
        input_variables=["input", "dialect", "table_info", "top_k", "proper_nouns"],
    )

    # Create the SQL agent using the full prompt and the database connection
    agent = create_sql_agent(
        llm=llm,
        db=db,
        extra_tools=[retriever_tool],
        prompt=few_shot_prompt,
        agent_type="openai-tools",
        verbose=True,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )

    # Invoke the agent to process the question and return the response
    res = agent.invoke({"input": question})
    for action, _ in res["intermediate_steps"]:
        for message in action.message_log:
            if message.content.strip():
                print(message.content)

    print(res['output'])
    return res['output']

# Stop tracing memory allocations and display top memory-consuming lines
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)

# To ensure Flask app runs only when executed directly
if __name__ == '__main__':
    app.run(debug=True)

# Take a snapshot
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 memory consuming lines ]")
for stat in top_stats[:10]:
    print(stat)

# Stop tracing memory allocations
tracemalloc.stop()
