import os
os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"

import sys
sys.path.insert(0, r"C:\Users\Tanmay\Research_assistant")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("rag-knowledge-base")

@mcp.tool()
def query_knowledge_base(question: str) -> str:
    """
    Search the knowledge base and answer questions about the documents.
    Use this when you need to find information from the uploaded documents.
    
    Args:
        question: The question to search for in the documents
    
    Returns:
        An answer based on the relevant document content
    """
    from rag import query  # ← load only when first called, not at startup
    return query(question)

if __name__ == "__main__":
    mcp.run()