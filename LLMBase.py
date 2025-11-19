import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class ReformatQueryResponse(BaseModel):
    reformatted_query: str = Field(description="The reformatted query string")
    rationale: str = Field(description="Explanation of the changes made to the original query")

class DocumentRerankingResponse(BaseModel):
    ranked_documents: list[str] = Field(description="List of documents ranked by relevance")
    rationale: str = Field(description="Explanation of the ranking decisions")

class AnswerGenerationResponse(BaseModel):
    answer: str = Field(description="The generated answer to the query")
    reference: str = Field(description="Evidence from the context that supports the answer")
    rationale: str = Field(description="Explanation of how the answer was generated")


# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

class LLMBase:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        if model_name == "gpt-4o-mini":
            self.llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        elif model_name == "gemini-2.5-flash":
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))
        else:
            raise ValueError(f"Invalid model name: {model_name}")
    
    def reformate_query(self, query: str, prompt_override: str | None = None) -> str:
        prompt = prompt_override or self.get_model_prompt("prompt_reformation")

        structured_llm = self.llm.with_structured_output(ReformatQueryResponse)

        response = structured_llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ])

        return response
    
    def rerank_documents(self, query: str, documents: list[str], prompt_override: str | None = None) -> list[str]:
        prompt_template = prompt_override or self.get_model_prompt("document_reranking")
        prompt = prompt_template.format(query=query)

        structured_llm = self.llm.with_structured_output(DocumentRerankingResponse)
        
        response = structured_llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n\n".join(documents)}
        ])

        return response
    
    # def synthesize_context(self, query: str, documents: list[str]) -> str:
    #     prompt = self.get_model_prompt("context_synthesize").format(query=query)

    #     structured_llm = self.llm.with_structured_output(ContextSynthesisResponse)

    #     response = structured_llm.invoke([
    #         {"role": "system", "content": prompt},
    #         {"role": "user", "content": "\n\n".join(documents)}
    #     ])

    #     return response
    
    def generate_answer(self, query: str, context: str, prompt_override: str | None = None) -> str:
        prompt_template = prompt_override or self.get_model_prompt("answer_generation")
        prompt = prompt_template.format(context=context, query=query)

        structured_llm = self.llm.with_structured_output(AnswerGenerationResponse)

        response = structured_llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ])

        return response 
    
    def get_model_prompt(self, prompt_type: str) -> str | None:
        prompts = {
            "prompt_reformation": """
            You are an expert at reformulating user queries for information retrieval.
            Your task is to enhance the query while preserving the original intent.

            Guidelines:
            - Add relevant technical terms and synonyms
            - Make the query more specific and focused
            - Optimize for both semantic and keyword matching
            - Preserve key concepts from the original query

            Reformulate the following query for better retrieval:
            """,
            "document_reranking": """
            You are an expert at evaluating document relevance for question answering.
            Your task is to rank documents by their relevance to the specific query.

            Ranking Criteria:
            - Documents with direct answers get highest priority
            - Comprehensive explanations rank second
            - Supporting examples and context rank third
            - Off-topic or tangential content ranks lowest

            Query: {query}

            Rank these documents by relevance (most relevant first):
            """,
            "context_synthesize": """
            You are an expert at synthesizing information from multiple documents.
            Your task is to create a comprehensive context that directly addresses the query.

            Guidelines:
            - Focus on information most relevant to the user's question
            - Integrate information from multiple sources seamlessly
            - Remove redundant or conflicting information
            - Maintain factual accuracy and important details

            Query: {query}

            Synthesize the following retrieved documents:
            """,
            "answer_generation": """
            You are an AI assistant providing expert-level answers.
            Your task is to generate accurate, comprehensive responses based on the provided context.

            Guidelines:
            - Base your answer primarily on the provided context
            - Structure your response with clear explanations
            - Include specific details and examples when available
            - If context is insufficient, acknowledge the limitation clearly

            Context: {context}

            Question: {query}

            Provide a thorough, accurate answer:
            """
        }
        return prompts.get(prompt_type, None)
