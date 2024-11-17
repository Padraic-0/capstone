#import getpass
import os
from dotenv import load_dotenv

#os.environ["OPENAI_API_KEY"] = getpass.getpass()
from langchain_openai import ChatOpenAI

load_dotenv("password.env")  # Loads environment variables from .env
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import requests
import json
import jsonschema





# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

retrieved_docs = rag_chain.invoke("What is Task Decomposition?")


question = "What is Task Decomposition?"
student_response = "Task decomposition breaks complex tasks into smaller steps, using techniques like Chain of Thought to enhance reasoning and explore options."

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"
}

json_schema = {
    "type": "object",
    "properties": {
        "evaluation": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "grade": {"type": "integer"},
                "feedback": {"type": "string"},
                "follow_up_questions": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["summary", "grade", "feedback", "follow_up_questions"]
        }
    },
    "required": ["evaluation"]
}

json_template = f"""{{
  "evaluation": {{
    "summary": "",  // brief evaluation text, 50 words max
    "grade": 0,     // integer score from 0-10
    "feedback": "", // any additional feedback on performance
    "follow_up_questions": [] // questions to guide student if grade < 8
  }}
}}"""

data = {
    "model": "gpt-4-turbo",
    "messages": [{"role": "user", "content": f"As a grader, compare the student response to the question {question} to the textbox definition as defined as \"{retrieved_docs}\". Limit your evaluation to 50 words. At the end return a grade score 0-10. If the grade is below an 8 ask the student a few questions based to the textbook definition to get them on the right track to answer the question better next time. Student response: \"{student_response}\". Structure your response to fit this template: {json_template}"}],
    "temperature": 0.7
}
print(question, "\n")
print(f'Student response: {student_response}\n')
print(f'Chat GPT Message: {data["messages"][0]["content"]}\n')

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

result = response.json()

if response.status_code == 200:
    
    try:
        json_object = json.loads(result["choices"][0]["message"]["content"])
        jsonschema.validate(instance=json_object, schema=json_schema)
        print("JSON object is valid!")
        print(f'Graded response (JSON OBJECT):\n{json.dumps(json_object, indent=4)}')
    except jsonschema.exceptions.ValidationError as err:
        print(f"JSON object is invalid: {err.message}")
    
else:
    print(f"Error: {response.status_code} - {response.json()}")

#print(retrieved_docs)