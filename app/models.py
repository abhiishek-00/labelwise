"""Shared data models for products, retrieval results, and the API."""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Any
 
from pydantic import BaseModel, Field
 
# Nutrition fields exposed for comparison, with display labels and units.
NUTRITION_FIELDS: list[tuple[str, str, str]] = [
     ("energy_kcal_100g", "Energy", "kcal"),
     ("fat_100g", "Fat", "g"),
     ("saturated_fat_100g", "Saturated fat", "g"),
     ("carbohydrates_100g", "Carbohydrates", "g"),
     ("sugars_100g", "Sugars", "g"),
     ("fiber_100g", "Fiber", "g"),
     ("proteins_100g", "Protein", "g"),
     ("salt_100g", "Salt", "g"),
 ]
 
 
@dataclass(slots=True)
class Product:
     """A product document from the knowledge base."""
 
     code: str
     product_name: str
     brands: str | None = None
     categories: str | None = None
     ingredients_text: str | None = None
     allergens: str | None = None
     labels: str | None = None
     countries: str | None = None
     serving_size: str | None = None
     nutriscore_grade: str | None = None
     energy_kcal_100g: float | None = None
     fat_100g: float | None = None
     saturated_fat_100g: float | None = None
     carbohydrates_100g: float | None = None
     sugars_100g: float | None = None
     fiber_100g: float | None = None
     proteins_100g: float | None = None
     salt_100g: float | None = None
     doc_text: str = ""
 
     @classmethod
     def from_row(cls, row: Any) -> "Product":
         data = dict(row)
         allowed = {f for f in cls.__slots__}
         return cls(**{k: v for k, v in data.items() if k in allowed})
 
     def nutrition(self) -> dict[str, float | None]:
         return {field_name: getattr(self, field_name) for field_name, _, _ in NUTRITION_FIELDS}
 
     def to_dict(self) -> dict[str, Any]:
         return {f: getattr(self, f) for f in self.__slots__}
 
 
@dataclass(slots=True)
class SearchResult:
     """One retrieved product with the scores that produced its ranking.
 
     Component scores are retained so evaluation can attribute a ranking to the
     lexical or vector side, rather than only observing the combined result.
     """
 
     product: Product
     score: float = 0.0
     lexical_score: float | None = None
     vector_score: float | None = None
     rerank_score: float | None = None
     rank: int = 0
 
     @property
     def code(self) -> str:
         return self.product.code
 
 
@dataclass(slots=True)
class RetrievalConfig:
     """Parameters defining one retrieval strategy.
 
     Every evaluated variant is expressible as an instance of this, so the
     benchmark and production share a single code path.
     """
 
     method: str = "hybrid" # lexical | vector | hybrid
     hybrid_strategy: str = "weighted" # weighted | rrf
     alpha: float = 0.5 # weight on lexical when strategy is weighted
     top_n: int = 20 # candidates retrieved before reranking
     top_k: int = 5 # documents passed to the LLM
     rerank: bool = False
     rewrite: bool = False
 
     def label(self) -> str:
         """Stable identifier used in evaluation output and monitoring."""
         if self.method == "hybrid":
             base = (
                 f"hybrid_rrf" if self.hybrid_strategy == "rrf" else f"hybrid_a{self.alpha:g}"
             )
         else:
             base = self.method
         if self.rerank:
             base += "+rerank"
         if self.rewrite:
             base += "+rewrite"
         return base
 
 
@dataclass(slots=True)
class RetrievalOutcome:
     """Results plus the timing and query transformations that produced them."""
 
     results: list[SearchResult] = field(default_factory=list)
     original_query: str = ""
     rewritten_query: str | None = None
     config_label: str = ""
     retrieval_time_ms: float = 0.0
     rerank_time_ms: float = 0.0
     rewrite_time_ms: float = 0.0
     rewrite_tokens: int = 0
     rerank_tokens: int = 0
 
     @property
     def codes(self) -> list[str]:
         return [r.code for r in self.results]
 
     @property
     def effective_query(self) -> str:
         return self.rewritten_query or self.original_query
 
 
 # ── API schemas ───────────────────────────────────────────────────────────────
 
 
class QueryRequest(BaseModel):
     query: str = Field(min_length=1, max_length=1000)
     history: list[str] = Field(
         default_factory=list,
         description="Previous user turns, used for conversational query rewriting.",
     )
     top_k: int | None = Field(default=None, ge=1, le=20)
 
 
class SourceOut(BaseModel):
     code: str
     product_name: str
     brands: str | None = None
     categories: str | None = None
     nutriscore_grade: str | None = None
     score: float
     nutrition: dict[str, float | None] = Field(default_factory=dict)
 
 
class QueryResponse(BaseModel):
     conversation_id: str
     question: str
     rewritten_question: str | None = None
     answer: str
     sources: list[SourceOut] = Field(default_factory=list)
     retrieval_method: str
     model: str
     llm_backend: str
     total_tokens: int
     estimated_cost_usd: float
     response_time_ms: float
     relevance: str | None = None
     grounding_score: float | None = None
 
 
class FeedbackRequest(BaseModel):
     conversation_id: str
     feedback: int = Field(description="+1 for helpful, -1 for unhelpful")
 
 
class CompareRequest(BaseModel):
     codes: list[str] = Field(min_length=2, max_length=5)

