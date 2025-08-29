from typing import List
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.agents import AgentExecutor
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from sqlalchemy.orm import Session

from langchain_experimental.sql.base import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from duckduckgo_search import DDGS
import os
import logging

from rag.agentic_rag.db import get_vector_store_index, get_engine

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# LangChain Agent Setup
# ----------------------------------------------------------------------------

def get_agent_instance(db: Session) -> AgentExecutor:
    if not hasattr(get_agent_instance, "_agent"):
        get_agent_instance._agent = create_agent(db)
    return get_agent_instance._agent


def create_agent(db: Session) -> AgentExecutor:
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    index = get_vector_store_index("li_document")
    query_engine = index.as_query_engine(similarity_top_k=10, show_progress=True)

    engine = get_engine()
    sql_db = SQLDatabase(engine)
    sql_chain = SQLDatabaseChain.from_llm(llm, sql_db, verbose=True)

    def vector_search_tool(query: str) -> str:
        try:
            response = query_engine.query(query)
            source_nodes = getattr(response, "source_nodes", [])
            logger.info(f"Response: {source_nodes}")
            for i, node in enumerate(source_nodes):
                logger.info(f"[Node {i}] Score: {node.score}, Text: {node.node.get_content()[:200]}")
            return str(response)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return "Vector search failed."

    def sql_query_tool(query: str) -> str:
        try:
            return sql_chain.run(query)
        except Exception as exc:
            return f"SQL error: {exc}"

    def web_search_tool(query: str) -> str:
        try:
            ddgs = DDGS()
            results = ddgs.text(keywords=query, max_results=5)
            if not results:
                return "No search results found."
            snippets = [res.get("body", "") or res.get("snippet", "") for res in results]
            return "\n".join(snippets)
        except Exception as exc:
            return f"Search error: {exc}"

    tools: List[Tool] = [
        Tool(name="vector_search", func=vector_search_tool, description="Search internal documents."),
        Tool(name="sql_query", func=sql_query_tool, description="Query the SQL database."),
        Tool(name="web_search", func=web_search_tool, description="Search the web when internal sources are insufficient.")
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        memory=memory,
    )


async def get_contextual_answer(question: str, db: Session) -> str:
    """
    End-to-end retrieval + reranking + synthesis + guardrails.
    """

    try:
        from llama_index.core.response_synthesizers import CompactAndRefine
        from llama_index.core.postprocessor.llm_rerank import LLMRerank
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
        LLAMAINDEX_AVAILABLE = True
    except Exception:
        LLAMAINDEX_AVAILABLE = False

    try:
        from FlagEmbedding import FlagReranker
        FLAG_RERANKER_AVAILABLE = True
    except Exception:
        FLAG_RERANKER_AVAILABLE = False

    try:
        from nemoguardrails import LLMRails, RailsConfig
        GUARDRAILS_AVAILABLE = True
    except Exception:
        GUARDRAILS_AVAILABLE = False

    openai_key = os.getenv("OPENAI_API_KEY")

    # ------------------------------------------------------------------
    # 1. Retrieval from PGVector
    # ------------------------------------------------------------------
    if not LLAMAINDEX_AVAILABLE:
        logger.info("LlamaIndex not available; falling back to LangChain agent.")
        agent = get_agent_instance(db)
        return agent.run(question)

    index = get_vector_store_index("li_document")
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(question)

    nodes = response.source_nodes or []
    logger.info(f"Retrieved {len(nodes)} documents from vector store.")
    logger.info(f"Retrieved Nodes: {nodes}")
    reranked_nodes = nodes

    # ------------------------------------------------------------------
    # 2. Rerank using OpenAI LLM reranker
    # ------------------------------------------------------------------
    try:
        logger.info(f"Applying LLM reranker to {len(nodes)} nodes.")
        ranker = LLMRerank(
            choice_batch_size=5,
            top_n=3,
            llm=LlamaOpenAI(model="gpt-3.5-turbo", api_key=openai_key),
        )
        reranked_nodes = ranker.postprocess_nodes(nodes, query_str=question)
        logger.info(f"LLM reranker selected {len(reranked_nodes)} nodes.")
        logger.info(f"LLM reranker selected nodes {reranked_nodes}")
    except Exception as exc:
        logger.warning(f"LLM reranker failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Rerank using FlagEmbedding (if available)
    # ------------------------------------------------------------------
    try:
        if FLAG_RERANKER_AVAILABLE and reranked_nodes:
            logger.info(f"Applying FlagEmbedding reranker to {len(reranked_nodes)} nodes.")
            if not hasattr(get_contextual_answer, "_flag_reranker"):
                get_contextual_answer._flag_reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
            reranker = get_contextual_answer._flag_reranker
            pairs = [[question, n.node.get_content()] for n in reranked_nodes]
            scores = reranker.compute_score(pairs)
            reranked_nodes = [
                n for _, n in sorted(zip(scores, reranked_nodes), key=lambda x: x[0], reverse=True)
            ][:3]
            logger.info(f"Flag reranker selected {len(reranked_nodes)} nodes.")
            logger.info(f"Flag reranker selected nodes {reranked_nodes}")
            
    except Exception as exc:
        logger.warning(f"FlagEmbedding reranker failed: {exc}")

    # ------------------------------------------------------------------
    # 4. Answer synthesis
    # ------------------------------------------------------------------
    try:
        synthesiser = CompactAndRefine(
            llm=LlamaOpenAI(model="gpt-3.5-turbo", api_key=openai_key),
            verbose=False,
        )
        response = synthesiser.synthesize(question, nodes=reranked_nodes)
        answer_text = "Meeting started by abc@abc.com " + response.response + " Finally meeting ended by bva@abc.com"

        logger.info("Synthesis complete.")
        logger.info(f"Synthesized answer {answer_text}")
    except Exception as exc:
        logger.warning(f"Synthesis failed: {exc}")
        answer_text = "\n\n".join([n.node.get_content() for n in reranked_nodes])

    # ------------------------------------------------------------------
    # 5. Guardrails (mask emails, async-safe)
    # ------------------------------------------------------------------
    import re
    EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

    def mask_emails(text: str) -> str:
        return EMAIL_RE.sub("[***]", text)

    if GUARDRAILS_AVAILABLE:
        from nemoguardrails.llm.types import Task
        try:
            config = RailsConfig.from_path("RailConfigPath")
            rails = LLMRails(config)
            rails.register_output_parser(mask_emails, name="mask_emails")

            guarded = await rails.generate_async(messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer_text},
            ])

            parsed = rails.runtime.llm_task_manager.parse_task_output(
                task=Task.GENERATE_BOT_MESSAGE,
                output=guarded["content"],
                forced_output_parser="mask_emails"
            )
            answer_text = parsed.text
            logger.info("Guardrails applied.")
            logger.info(f"Guardrails output {answer_text}")
        except Exception as exc:
            logger.warning(f"Guardrails failed: {exc}")
            # minimal fallback
            answer_text = mask_emails(answer_text)
    else:
        answer_text = mask_emails(answer_text)

    return answer_text

    



