"""
Base Module Interface for RAG Pipeline

Each module in the RAG pipeline must:
1. Be independently testable
2. Have clear input/output contracts
3. Support prompt injection for GEPA optimization
4. Be serializable for logging and evaluation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TypeVar, Generic
from enum import Enum
import json
from pathlib import Path


class ModuleType(Enum):
    """Module identifiers for the RAG pipeline"""
    QUERY_PLANNER = "query_planner"
    RETRIEVAL = "retrieval"
    RERANKER = "reranker"
    GENERATOR = "generator"


@dataclass
class ModuleInput:
    """Base class for module inputs - all inputs must be serializable"""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleInput":
        return cls(**data)


@dataclass
class ModuleOutput:
    """Base class for module outputs - all outputs must be serializable"""
    status: str = "success"  # "success" or "error"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModuleOutput":
        return cls(**data)


# Generic types for module I/O
InputT = TypeVar("InputT", bound=ModuleInput)
OutputT = TypeVar("OutputT", bound=ModuleOutput)


class Module(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all RAG pipeline modules.

    Each module must:
    1. Implement run() for execution
    2. Support prompt injection/restoration for GEPA
    3. Be independently evaluable
    """

    def __init__(self, module_type: ModuleType):
        self.module_type = module_type
        self._prompt: Optional[str] = None
        self._original_prompt: Optional[str] = None

    @abstractmethod
    async def run(self, input: InputT) -> OutputT:
        """
        Execute the module with the given input.

        Args:
            input: Module-specific input

        Returns:
            Module-specific output
        """
        pass

    @property
    def prompt(self) -> Optional[str]:
        """Get the current prompt"""
        return self._prompt

    @prompt.setter
    def prompt(self, value: str) -> None:
        """Set the prompt (used by GEPA optimization)"""
        if self._original_prompt is None:
            self._original_prompt = self._prompt
        self._prompt = value

    def restore_prompt(self) -> None:
        """Restore the original prompt after GEPA optimization"""
        if self._original_prompt is not None:
            self._prompt = self._original_prompt
            self._original_prompt = None

    def save_prompt(self, path: Path, version: str = "v1") -> None:
        """Save the current prompt to a file"""
        prompt_path = path / f"{version}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(self._prompt or "")

    def load_prompt(self, path: Path) -> None:
        """Load a prompt from a file"""
        if path.exists():
            self._prompt = path.read_text()

    @abstractmethod
    def get_default_prompt(self) -> str:
        """Return the default/seed prompt for this module"""
        pass


@dataclass
class EvaluationResult:
    """Result of module evaluation"""
    module_type: ModuleType
    metrics: Dict[str, float]
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    prompt_version: str = "baseline"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["module_type"] = self.module_type.value
        return result


class ModuleEvaluator(ABC):
    """Abstract base class for module-specific evaluators"""

    @abstractmethod
    async def evaluate(
        self,
        module: Module,
        input_data: ModuleInput,
        output_data: ModuleOutput,
        ground_truth: Optional[Any] = None
    ) -> EvaluationResult:
        """
        Evaluate module output against ground truth.

        Args:
            module: The module being evaluated
            input_data: Input that was passed to the module
            output_data: Output from the module
            ground_truth: Optional ground truth for comparison

        Returns:
            EvaluationResult with metrics
        """
        pass


# ============================================================================
# Module-Specific Input/Output Types
# ============================================================================

@dataclass
class QueryPlannerInput(ModuleInput):
    """Input for Query Planner module"""
    query: str
    feedback: Optional[str] = None


@dataclass
class QueryPlannerOutput(ModuleOutput):
    """Output from Query Planner module"""
    mode: str = "reformulation"  # "decomposition" or "reformulation"
    queries: List[str] = field(default_factory=list)
    original_query: str = ""


@dataclass
class RetrievalInput(ModuleInput):
    """Input for Retrieval module"""
    queries: List[str]  # From query planner
    top_k: int = 20


@dataclass
class RetrievalOutput(ModuleOutput):
    """Output from Retrieval module"""
    documents: List[Dict[str, Any]] = field(default_factory=list)
    document_texts: List[str] = field(default_factory=list)
    chunk_indices: List[int] = field(default_factory=list)
    retrieved_doc_ids: List[str] = field(default_factory=list)  # For document-level evaluation


@dataclass
class RerankerInput(ModuleInput):
    """Input for Reranker module"""
    query: str
    documents: List[str]  # Document texts to rerank
    feedback: Optional[str] = None


@dataclass
class RerankerOutput(ModuleOutput):
    """Output from Reranker module"""
    ranked_documents: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)


@dataclass
class GeneratorInput(ModuleInput):
    """Input for Generator module"""
    query: str
    context: str  # Concatenated relevant documents
    feedback: Optional[str] = None


@dataclass
class GeneratorOutput(ModuleOutput):
    """Output from Generator module"""
    answer: str = ""
    reference: str = ""
    rationale: str = ""


# ============================================================================
# Pipeline Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline"""
    # Model settings
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Retrieval settings
    retrieval_k: int = 20
    final_k: int = 10
    chunk_size: int = 600
    chunk_overlap: int = 50

    # Evaluation settings
    quality_threshold: float = 0.80

    # Prompt paths
    prompt_dir: Path = field(default_factory=lambda: Path("modules"))

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["prompt_dir"] = str(self.prompt_dir)
        return result
