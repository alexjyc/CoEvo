import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class QueryPlannerResponse(BaseModel):
    mode: str = Field(description="The mode of the query planner. Either 'decomposition' or 'reformulation'.")
    query: str | list[str] = Field(description="The query to be planned. If the mode is 'decomposition', it is a list of sub-queries. If the mode is 'reformulation', it is a single query.")

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
        if model_name in ["gpt-4o-mini", "gpt-4o", "gpt-5-nano"]:
            self.llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        elif model_name == "gemini-2.5-flash":
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))
        else:
            raise ValueError(f"Invalid model name: {model_name}")
    
    async def query_planner(self, query: str, feedback: str=None) -> QueryPlannerResponse:
        """Async query planning for decomposition or reformulation"""
        prompt = self.get_model_prompt("query_planner")
        prompt = prompt.format(feedback=feedback if feedback else "")

        structured_llm = self.llm.with_structured_output(QueryPlannerResponse)

        response = await structured_llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ])

        return response
    
    async def rerank_documents(self, query: str, documents: list[str], feedback: str=None) -> DocumentRerankingResponse:
        """Async listwise reranking of all documents by relevance"""
        prompt = self.get_model_prompt("document_reranking")
        
        # Number the documents for clear reference
        numbered_docs = [f"[{i+1}] {doc}" for i, doc in enumerate(documents)]
        documents_text = "\n\n".join(numbered_docs)
        
        prompt = prompt.format(
            query=query, 
            feedback=feedback if feedback else "",
            num_documents=len(documents)
        )

        structured_llm = self.llm.with_structured_output(DocumentRerankingResponse)
        
        response = await structured_llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": documents_text}
        ])

        return response
    
    async def generate_answer(self, query: str, context: str, feedback: str=None) -> AnswerGenerationResponse:
        """Async answer generation from context"""
        prompt = self.get_model_prompt("answer_generation")
        prompt = prompt.format(context=context, query=query, feedback=feedback if feedback else "")

        structured_llm = self.llm.with_structured_output(AnswerGenerationResponse)

        response = await structured_llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ])

        return response 
    
    def get_model_prompt(self, prompt_type: str) -> str | None:
        prompts = {
            "query_planner": """
            You are an expert in information-retrieval query handling for a hybrid search pipeline that uses BM25 (sparse keyword search) and FAISS (dense semantic search).
            Your task is to decide—based on the complexity of the user’s request—whether to:
            - Decompose the query into a minimal set of targeted sub-queries
            OR
            - Reformulate the query into an enhanced single retrieval query.

            Decision Logic:
            - If the user request contains multiple intents, steps, constraints, or data types → perform Query Decomposition.
            - If the user request is single-intent but vague, ambiguous, or underspecified → perform Query Reformulation.

            Query Decomposition Guidelines:
            - Break the user’s request into the minimal set of sub-queries needed to fully satisfy the information need.
            - Preserve the original intent; do not invent new goals.
            - Identify key entities, time spans, modalities, and constraints.
            - Produce ordered, specific sub-queries, each with a brief rationale.
            - Ensure the set of sub-queries collectively covers the whole task without redundancy.

            Query Reformulation Guidelines:
            - Enhance the retrieval quality while preserving the intent.
            - Add relevant technical synonyms, expansions, disambiguations, and specificity.
            - Optimize for both keyword and semantic matching.
            - Keep it as a single improved query, not multiple.

            Feedback (if any): 
            {feedback}

            Provide a mode and query:
            """,
            "document_reranking": """
            You are an expert at evaluating document relevance for question answering.
            Your task is to RERANK ALL {num_documents} documents by their relevance to the query.
            
            CRITICAL INSTRUCTIONS:
            1. You will receive {num_documents} numbered documents [1], [2], [3], etc.
            2. You MUST return ALL {num_documents} documents in your ranked list
            3. Reorder them from most relevant (first) to least relevant (last)
            4. Return ONLY the complete document texts in the new order
            5. Do NOT return document numbers, just the full text of each document

            Ranking Criteria:
            - Documents with direct answers get highest priority
            - Comprehensive explanations rank second
            - Supporting examples and context rank third
            - Off-topic or tangential content ranks lowest
            
            Feedback (if any): 
            {feedback}

            Query: {query}

            Rerank these {num_documents} documents by relevance (most relevant first, return ALL documents):
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
