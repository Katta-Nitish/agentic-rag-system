import streamlit as st
import time
import json

from prompts import QUERY_REWRITER_SYSTEM_PROMPT, QUERY_ANALYZER_SYSTEM_PROMPT, EVALUATOR_SYSTEM_PROMPT, RESPONSE_GENERATOR_SYSTEM_PROMPT

from langgraph.graph import StateGraph, START,END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.messages import SystemMessage
import numexpr as ne
import arxiv
from langchain_experimental.tools import PythonREPLTool
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker
)

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever
)
from langgraph.types import interrupt, Command
from langchain_ollama.embeddings import OllamaEmbeddings

from typing import TypedDict, Optional

if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = InMemorySaver()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(time.time())

st.set_page_config(page_title="Agentic RAG System", page_icon="🤖")
st.title("Agentic Retrieval-Augmented Generation System")
st.markdown(
    """
    <p style='font-size:18px; color:#A0A0A0; margin-top:-10px;'>
    An evaluator-driven LangGraph workflow with retrieval routing,
    ambiguity-aware clarification, tool orchestration,
    and grounded response generation.
    </p>
    """,
    unsafe_allow_html=True
)

class State(TypedDict):
    query: str
    formatted_query: str

    analysis_result: dict
    action: str

    needs_clarification: bool
    needs_retrieval: bool
    needs_tool: bool

    clarification_question: Optional[str]

    tool_name: Optional[str]
    tool_input: Optional[str]

    retrieved_context: Optional[list]

    arxiv_results: Optional[list]
    calculation_result: Optional[str]
    code_execution_result: Optional[str]

    evidence_sufficient: Optional[bool]
    evaluation_reasoning: Optional[str]
    failure_reason: Optional[str]

    generated_response: Optional[str]

def query_rewriter(state: State):
    llm=ChatOllama(model="qwen3:4b", temperature=0.7, keep_alive=True, base_url="http://host.docker.internal:11434")

    prompt=ChatPromptTemplate.from_messages([
        SystemMessage(content=QUERY_REWRITER_SYSTEM_PROMPT),
        ("human", "{query}"),
    ])
    chain=prompt | llm | StrOutputParser()
    return {"formatted_query": chain.invoke({"query": state["query"]})}



def query_analyzer(state: State):
    llm=ChatOllama(model="gemma3:12b", temperature=0.7,format="json", keep_alive=True, base_url="http://host.docker.internal:11434")    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=QUERY_ANALYZER_SYSTEM_PROMPT),
        ("user", "{query}")
    ])

    chain=prompt | llm | StrOutputParser()
    response=chain.invoke({"query": state['formatted_query']})
    try:
        result = json.loads(response)

    except Exception:
        return {
            "action": "REFUSE",
            "needs_retrieval": False,
            "needs_tool": False,
            "needs_clarification": False,
            "tool_name": None,
            "tool_input": None,
            "clarification_question": None,
            "failure_reason": "INVALID_ANALYZER_OUTPUT"
        }
    
    tool_name = result.get("tool_name")
    if tool_name in ["Python_Code_Execution_Tool", "Python_code_execution_Tool"]:
        tool_name = "python_code_execution_tool"
    elif tool_name in ["Calculator_Tool", "calculator_Tool"]:
        tool_name = "calculator_tool"
    elif tool_name in ["arXiv_Search_Tool", "arXiv_search", "arxiv_search_tool"]:
        tool_name = "arxiv_search"
    
    return {"analysis_result": result, "needs_clarification": bool(result.get("needs_clarification", False)), "needs_tool": bool(result.get("needs_tool", False)), "tool_name": tool_name, "tool_input": result.get("tool_input"), "action": result.get("action"), "needs_retrieval": bool(result.get("needs_retrieval", False)), "clarification_question": result.get("clarification_question")}

def clarifier(state: State):
    question=state.get("clarification_question")

    clarification=interrupt(question)

    updated_response=f"Original Query: {state['query']}\nClarification: {clarification}"
    return {"query": updated_response}



@st.cache_resource
def load_cached_vector():
    return FAISS.load_local("faiss_index", OllamaEmbeddings(model="nomic-embed-text", base_url="http://host.docker.internal:11434"), allow_dangerous_deserialization=True)

def retriver(state: State):
    vector=load_cached_vector()
    retriever=vector.as_retriever(search_kwargs={"k": 20})
    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor=CrossEncoderReranker(model=model, top_n=6)
    compression_retriver=ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    reranker_doc=compression_retriver.invoke(state['formatted_query'])
    docs=[
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in reranker_doc
    ]
    return {"retrieved_context": docs}


def arXiv_search(state: State):
    client=arxiv.Client()
    search = arxiv.Search(
        query=f"cat:cs.AI AND {state['tool_input']}",
        max_results=5,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers=[]
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "url": result.entry_id
        })
    return {"arxiv_results": papers}

def calculator(state: State):
    expression= state['tool_input']
    try:
        result = ne.evaluate(expression)
        return {"calculation_result": str(result)}

    except Exception as e:
        return {
            "calculation_result":
            f"Calculation Error: {str(e)}"
        }

def code_execution(state: State):
    code= state['tool_input']
    runner=PythonREPLTool()
    try:
        result=runner.run(code)
        return {"code_execution_result": repr(result)}
    except Exception as e:
        return {
            "code_execution_result":
            f"Code Execution Error: {str(e)}"
        }


def evaluator(state: State):
    llm=ChatOllama(model="llama3:8b", temperature=0.7,format="json", keep_alive=True, base_url="http://host.docker.internal:11434")
    tool_result = None

    if state.get("arxiv_results"):
        tool_result = state["arxiv_results"]

    elif state.get("calculation_result"):
        tool_result = state["calculation_result"]

    elif state.get("code_execution_result"):
        tool_result = state["code_execution_result"]

    tool_context = (
        {
            "tool_name": state.get("tool_name"),
            "tool_result": tool_result
        }
        if tool_result else None
    )
    retrieved_context = state.get("retrieved_context", []) if state["needs_retrieval"] else None
    direct_answer_context = state.get("action") if state.get("action") == "DIRECT_ANSWER" else None
    system_prompt= EVALUATOR_SYSTEM_PROMPT

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        ("human", """
        User Query:
        {query}
        Retrieved Context:
        {retrieved_context}
        Tool Context:
        {tool_context}
        Direct Answer Context:
        {direct_answer_context}
        """)
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": state["formatted_query"], "retrieved_context": retrieved_context, "tool_context": tool_context, "direct_answer_context": direct_answer_context})
    try:
        result = json.loads(response)

    except Exception:
        return {
            "action": "REFUSE",
            "needs_retrieval": False,
            "needs_tool": False,
            "needs_clarification": False,
            "tool_name": None,
            "tool_input": None,
            "clarification_question": None,
            "failure_reason": "INVALID_ANALYZER_OUTPUT"
        }
    return {"evaluation_reasoning": result.get("reasoning"), "evidence_sufficient": bool(result.get("evidence_sufficient", False)), "failure_reason": result.get("failure_reason")}

def response_generator(state: State):

    llm = ChatOllama(
        model="qwen3.5:9b",
        temperature=0,
        keep_alive=True,
        base_url="http://host.docker.internal:11434"
    )


    tool_data = None

    if state.get("tool_name") == "arxiv_search":
        tool_data = state.get("arxiv_results")

    elif state.get("tool_name") == "calculator_tool":
        tool_data = state.get("calculation_result")

    elif state.get("tool_name") == "python_code_execution_tool":
        tool_data = state.get("code_execution_result")

    formatted_tool_data = ""

    if state.get("tool_name") == "arxiv_search":

        papers = state.get("arxiv_results", [])

        formatted_tool_data = "\n\n".join([
            f"""
    Title: {paper.get('title')}

    Summary: {paper.get('summary')}

    URL: {paper.get('url')}
    """
            for paper in papers
        ])

    elif state.get("tool_name") == "calculator_tool":

        formatted_tool_data = str(
            state.get("calculation_result")
        )

    elif state.get("tool_name") == "python_code_execution_tool":

        formatted_tool_data = str(
            state.get("code_execution_result")
        )


    if state["action"] == "DIRECT_ANSWER":

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Answer conversationally and concisely.
                """
            ),
            ("human", "{query}")
        ])

        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "query": state["query"]
        })

        return {
            "generated_response": response
        }


    human_prompt = """
    User Query:
    {query}

    Action:
    {action}

    Tool Used:
    {tool_name}

    Tool Results:
    {tool_data}

    Retrieved Context:
    {retrieved_context}

    Evidence Evaluation:
    {evaluation_reasoning}
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=RESPONSE_GENERATOR_SYSTEM_PROMPT),
        ("human", human_prompt)
    ])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({

        "query": state["formatted_query"],

        "action": str(state.get("action")),

        "tool_name": str(state.get("tool_name")),

        "tool_data": formatted_tool_data,

        "retrieved_context": str(
            state.get("retrieved_context", [])
        ),

        "evaluation_reasoning": str(
            state.get("evaluation_reasoning", "")
        )
    })

    return {
        "generated_response": response
    }


def tool_dispatch(state: State):
    return {}


def route_after_analysis(state: State):

    action = str(state.get("action", "")).upper()

    if action == "CLARIFY":
        return "clarifier"

    elif action == "RETRIEVE":
        return "retriever"

    elif action == "TOOL":
        return "tool_dispatch"

    elif action == "DIRECT_ANSWER":
        return "response_generator"

    elif action == "REFUSE":
        return "refusal_node"


def route_after_evaluation(state: State):

    if state.get("evidence_sufficient", False):
        return "response_generator"

    return "refusal_node"


def tool_router(state: State):

    tool = state.get("tool_name")

    if tool == "arxiv_search":
        return "arxiv_search"

    elif tool == "calculator_tool":
        return "calculator"

    elif tool == "python_code_execution_tool":
        return "code_execution"

    return "refusal_node"


def refusal_node(state: State):

    action = state.get("action")
    failure_reason = state.get(
        "failure_reason",
        "INSUFFICIENT_CONTEXT"
    )

    query = state.get("query", "")

    if action == "REFUSE":

        return {
            "generated_response":
            (
                "I am designed primarily for AI research, "
                "RAG systems, and technical reasoning tasks. "
                "This query falls outside my supported scope."
            )
        }

    if failure_reason == "INSUFFICIENT_CONTEXT":

        return {
            "generated_response":
            (
                "I do not have sufficient grounded evidence "
                "to answer this reliably."
            )
        }

    elif failure_reason == "LOW_RELEVANCE":

        return {
            "generated_response":
            (
                "The retrieved information was not sufficiently "
                "relevant to answer the query reliably."
            )
        }

    elif failure_reason == "CONTRADICTORY_CONTEXT":

        return {
            "generated_response":
            (
                "The retrieved evidence contains conflicting "
                "information, so I cannot answer confidently."
            )
        }

    return {
        "generated_response":
        (
            "I cannot answer this reliably based on the "
            "available evidence."
        )
    }


def graph_builder():

    builder = StateGraph(State)

    builder.add_node("query_rewriter", query_rewriter)
    builder.add_node("query_analyzer", query_analyzer)

    builder.add_node("clarifier", clarifier)

    builder.add_node("retriever", retriver)

    builder.add_node("tool_dispatch", tool_dispatch)

    builder.add_node("arxiv_search", arXiv_search)
    builder.add_node("calculator", calculator)
    builder.add_node("code_execution", code_execution)

    builder.add_node("evaluator", evaluator)

    builder.add_node("response_generator", response_generator)
    builder.add_node("refusal_node", refusal_node)

    builder.add_edge(START, "query_rewriter")
    builder.add_edge("query_rewriter", "query_analyzer")

    builder.add_conditional_edges(
        "query_analyzer",
        route_after_analysis
    )

    builder.add_edge("clarifier", "query_rewriter")

    builder.add_conditional_edges(
        "tool_dispatch",
        tool_router
    )

    builder.add_edge("retriever", "evaluator")

    builder.add_edge("arxiv_search", "response_generator")
    builder.add_edge("calculator", "response_generator")
    builder.add_edge("code_execution", "response_generator")

    builder.add_conditional_edges(
        "evaluator",
        route_after_evaluation
    )

    builder.add_edge("response_generator", END)
    builder.add_edge("refusal_node", END)

    return builder.compile(
        checkpointer=st.session_state.agent_memory
    )

graph=graph_builder()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.pending_question:
    clarification = st.chat_input(st.session_state.pending_question)

    if clarification:
        with st.chat_message("user"):
            st.markdown(clarification)
        st.session_state.messages.append({"role": "user", "content": clarification})
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                result = graph.invoke(
                    Command(resume=clarification),
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        }
                    }
                )
                response = result.get(
                    "generated_response",
                    "No response generated."
                )
                st.markdown(response)
                with st.expander("Detailed Reasoning and Observations"):
                    st.markdown(f"""
                    Action Taken: {result.get('action', 'N/A')}\n
                    Tool Used: {result.get('tool_name', 'N/A')}\n
                    Tool Input: {result.get('tool_input', 'N/A')}\n
                    Retrieval Performed: {result.get('needs_retrieval', False)}\n
                    Evaluation Reasoning: {result.get('evaluation_reasoning', 'N/A')}\n
                    """)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.pending_question = None
        st.rerun()
elif prompt:=st.chat_input("Enter your research question here..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            result = graph.invoke(
                {"query": prompt},
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                }
            )
            if "__interrupt__" in result:

                question = result["__interrupt__"][0].value

                st.session_state.pending_question = question

                clarification_message = (
                    f"Clarification required before continuing.\n\n{question}"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": clarification_message
                })

                st.rerun()
            else:
                response = result.get(
                    "generated_response",
                    "No response generated."
                )

                st.markdown(response)
                with st.expander("Detailed Reasoning and Observations"):
                    st.markdown(f"""
                    Action Taken: {result.get('action', 'N/A')}\n
                    Tool Used: {result.get('tool_name', 'N/A')}\n
                    Tool Input: {result.get('tool_input', 'N/A')}\n
                    Retrieval Performed: {result.get('needs_retrieval', False)}\n
                    Evaluation Reasoning: {result.get('evaluation_reasoning', 'N/A')}\n
                    """)
                st.session_state.messages.append({"role": "assistant", "content": response})
