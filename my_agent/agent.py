import warnings
import os
from typing import List, Literal
from typing_extensions import TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Verify environment variables exist
assert os.getenv('ANTHROPIC_API_KEY'), "ANTHROPIC_API_KEY is required"
assert os.getenv('TAVILY_API_KEY'), "TAVILY_API_KEY is required" 
assert os.getenv('VOYAGE_API_KEY'), "VOYAGE_API_KEY is required"

# Initialize embeddings for vector store
embd = VoyageAIEmbeddings(
    voyage_api_key=os.environ['VOYAGE_API_KEY'],
    model="voyage-law-2"
)

# Define source URLs for knowledge base
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load and process documents from URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents into chunks for vector store
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Create vector store and retriever
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embd,
)
retriever = vectorstore.as_retriever()

# Define data models for routing and grading
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess if answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]

# Initialize LLM and tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
structured_llm_router = llm.with_structured_output(RouteQuery)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
structured_llm_hallucination = llm.with_structured_output(GradeHallucinations)
structured_llm_answer = llm.with_structured_output(GradeAnswer)
web_search_tool = TavilySearchResults(k=3)

# Define prompts for different components
route_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. Otherwise, use web-search."""),
    ("human", "{question}"),
])

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing whether an answer addresses / resolves a question.
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
])

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """You a question re-writer that converts an input question to a better version that is optimized
    for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
])

# Initialize chain components
question_router = route_prompt | structured_llm_router
retrieval_grader = grade_prompt | structured_llm_grader
hallucination_grader = hallucination_prompt | structured_llm_hallucination
answer_grader = answer_prompt | structured_llm_answer
question_rewriter = rewrite_prompt | llm | StrOutputParser()
rag_chain = hub.pull("rlm/rag-prompt") | llm | StrOutputParser()

# Node functions
def retrieve(state):
    """
    Retrieve documents from vector store
    
    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer based on retrieved documents
    
    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    
    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    Transform the query to produce a better question.
    
    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state):
    """
    Web search based on the re-phrased question.
    
    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "question": question}

# Edge functions
def route_question(state):
    """
    Route question to web search or RAG.
    
    Args:
        state (dict): The current graph state
    
    Returns:
        str: Next node to call
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    print("---ROUTE QUESTION TO RAG---")
    return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    
    Args:
        state (dict): The current graph state
    
    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    
    if not filtered_documents:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    print("---DECISION: GENERATE---")
    return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    
    Args:
        state (dict): The current graph state
    
    Returns:
        str: Decision for next node to call
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    
    if score.binary_score == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"
    print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
    return "not supported"

# Build graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Add edges
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile graph
app = workflow.compile()
graph = app