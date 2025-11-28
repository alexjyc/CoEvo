import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class QueryPlannerResponse(BaseModel):
    sub_queries: list[str] = Field(description="List of decomposed sub-queries for retrieval. Each sub-query targets a specific aspect of the original query.")

class DocumentRerankingResponse(BaseModel):
    ranked_documents: list[str] = Field(description="List of documents ranked by relevance")

class AnswerGenerationResponse(BaseModel):
    answer: str = Field(description="The generated answer to the query")
    reference: str = Field(description="Evidence from the context that supports the answer")
    rationale: str = Field(description="Explanation of how the answer was generated")


# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

class LLMBase:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        if model_name == "gpt-5-nano":
            self.llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        elif model_name == "gpt-4o-mini":
            self.llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        elif model_name == "gemini-2.5-flash":
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))
        else:
            raise ValueError(f"Invalid model name: {model_name}")
    
    def query_planner(self, query: str) -> list[str]:
        prompt = self.get_model_prompt("query_planner")

        structured_llm = self.llm.with_structured_output(QueryPlannerResponse)

        response = structured_llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ])

        return response
    
    def rerank_documents(self, query: str, documents: list[str]) -> list[str]:
        prompt = self.get_model_prompt("document_reranking")
        prompt = prompt.format(query=query)

        structured_llm = self.llm.with_structured_output(DocumentRerankingResponse)
        
        response = structured_llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n\n".join(documents)}
        ])

        return response
    
    def generate_answer(self, query: str, context: str) -> str:
        prompt = self.get_model_prompt("answer_generation")
        prompt = prompt.format(context=context, query=query)

        structured_llm = self.llm.with_structured_output(AnswerGenerationResponse)

        response = structured_llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ])

        return response 
    
    def get_model_prompt(self, prompt_type: str) -> str | None:
        prompts = {
            "query_planner": """
            You are an expert at user queries for information retrieval that combines BM25 (keyword matching) and FAISS (dense semantic search).
            Your task is to decompose the user's request into the minimal set of targeted sub-queries that, when retrieved, cover the entire information need.

            Guidelines:
            - Preserve the original intent; do not rewrite answers.
            - Identify key entities, time ranges, data modalities, and constraints.
            - Split multi-part requests into ordered sub-queries, each with a short rationale explaining why it is needed.
            - Prefer specific phrasing (e.g., include product names, document sections, time frames) over general descriptions.
            - Ensure the combined sub-queries jointly answer the overall task without redundancy.

            Feedback (if any): 
            {feedback}

            Return the decomposition as a numbered list of sub-queries with optional rationale per line.
            """,
            "document_reranking": """
            You are an expert at evaluating document relevance for question answering.
            Your task is to rank documents by their relevance to the specific query.

            Ranking Criteria:
            - Documents with direct answers get highest priority
            - Comprehensive explanations rank second
            - Supporting examples and context rank third
            - Off-topic or tangential content ranks lowest
            
            Feedback (if any): 
            {feedback}

            Query: {query}

            Rank these documents by relevance (most relevant first):
            """,
            "answer_generation": """
            You are an AI assistant providing expert-level answers.
            Your task is to generate accurate, comprehensive responses based on the provided context.

            Guidelines:
            - Base your answer primarily on the provided context
            - Structure your response with clear explanations
            - Include specific details and examples when available
            - If context is insufficient, acknowledge the limitation clearly

            Feedback (if any): 
            {feedback}

            Context: {context}

            Question: {query}

            Provide a thorough, accurate answer:
            """
        }
        return prompts.get(prompt_type, None)
