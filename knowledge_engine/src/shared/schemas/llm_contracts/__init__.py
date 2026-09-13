"""Pydantic v2 контракты для Gemini structured JSON (единый реестр)."""

from knowledge_engine.src.adapters.search_providers.exa_search import (
    BatchDomainAuthorityResponse,
    DomainAuthorityItem,
    DomainAuthorityVerdict,
    ExaSearchContextExpansion,
)
from knowledge_engine.src.domains.curriculum.contracts.curriculum import (
    CurriculumReasonerContract,
    ExpansionVectorContract,
    FlashCurriculumPayloadContract,
    FlashExpansionPatchContract,
    GeminiSourcesEnrichmentContract,
    ModelFirstPayloadContract,
)
from knowledge_engine.src.domains.curriculum.contracts.lite_curriculum import (
    ArxivQueryParamsContract,
    LiteAcademicQueryContract,
    LiteBatchEvalContract,
    LiteQueryPlanContract,
    LiteSiteSuggestionsContract,
    LiteSourceBatchContract,
)
from knowledge_engine.src.domains.curriculum.domain import DomainProfilerBatchContract
from knowledge_engine.src.domains.curriculum.research_schemas import (
    ClarificationConstraintsResponse,
    HarvestedLinksResponse,
    ReplFollowUpResponse,
)
from knowledge_engine.src.domains.curriculum.source_eval import (
    SourceEvaluatorLiteContract,
)
from knowledge_engine.src.domains.grounding.drill_schemas import (
    ActiveDrillStepResponse,
    AnswerAccuracyGrade,
    DrillStepResponse,
    LayerCompletionTutorOutput,
    StandardDrillTutorOutput,
    TechnicalConceptAudit,
)
from knowledge_engine.src.domains.grounding.evaluator_critique import (
    EvaluatorCritiqueContract,
)
from knowledge_engine.src.domains.grounding.tutor import (
    STRUCTURED_LECTURE_FIELD_RULES,
    DeepDiveDeepAnalysisContract,
    DeepDiveExplainContract,
    DeepDiveTutorContract,
    DialogueFactManifestContract,
    IntroAssessmentContract,
    NodeExplainContract,
    StepAnalysisContract,
    StructuredLectureResponse,
    SubConceptGapEvalContract,
    structured_lecture_to_dense,
)
from knowledge_engine.src.domains.ingestion.code_ingest.tiered_code_pruner import (
    TieredClassificationResult,
)
from knowledge_engine.src.domains.ingestion.paper_quality.paper_structure_schema import (
    PaperCredibilityAnalysis,
    PaperStructureAnalysis,
)
from knowledge_engine.src.domains.ingestion.pre_map_dedup import CanonicalMapContract
from knowledge_engine.src.legacy.unraveling_schemas import UnravelingNodeResponse
from knowledge_engine.src.legacy.v04_gemini import (
    AnalysisReportContract,
    GeminiL0DecompositionContract,
    L2EvidenceExtractionContract,
    ResearchEvaluationContract,
)
from knowledge_engine.src.shared.consensus import (
    AcademicQueryContract,
    ProfileApplicabilityContract,
    RefinementSanitizeContract,
    ValidationResultContract,
)
from knowledge_engine.src.shared.reasoner import FinalResponseContract
from knowledge_engine.src.shared.schemas.llm_contracts.analytics_v07 import (
    ChunkExtractionContract,
    ConceptGraphContract,
    ProfileGapMapContract,
    TradeoffMatrixContract,
)
from knowledge_engine.src.shared.vlm import VlmBatchResponseContract

# Реестр label → контракт (для trace / документации)
GEMINI_STRUCTURED_CONTRACTS: dict[str, type] = {
    "node_deep_dive / intro_assessment": IntroAssessmentContract,
    "node_deep_dive / dense_material": StructuredLectureResponse,
    "node_deep_dive/tutor": DeepDiveTutorContract,
    "node_deep_dive / tutor_explain": DeepDiveExplainContract,
    "node_deep_dive / drill_active": ActiveDrillStepResponse,
    "node_deep_dive / drill_complete": LayerCompletionTutorOutput,
    "node_deep_dive / step_analysis": StepAnalysisContract,
    "node_deep_dive / fact_manifest": DialogueFactManifestContract,
    "node_deep_dive / sub_concept_gap": SubConceptGapEvalContract,
    "node_deep_dive / deep_analysis_eval": EvaluatorCritiqueContract,
    "node_deep_dive / node_explain": NodeExplainContract,
    "node_selection_explain": NodeExplainContract,
    "contextual_explainer": NodeExplainContract,
    "curriculum_generator / draft": CurriculumReasonerContract,
    "curriculum_generator / search_first": FlashCurriculumPayloadContract,
    "curriculum_generator / model_first": ModelFirstPayloadContract,
    "curriculum / lite_sources_enrichment": GeminiSourcesEnrichmentContract,
    "curriculum / lite_expansion_vector": ExpansionVectorContract,
    "curriculum / flash_expansion_patch": FlashExpansionPatchContract,
    "reasoner / final": FinalResponseContract,
    "v04 / decomposition": GeminiL0DecompositionContract,
    "v04 / l2_extract": L2EvidenceExtractionContract,
    "v04 / research_eval": ResearchEvaluationContract,
    "v04 / matrix": AnalysisReportContract,
    "v04 / unraveling": UnravelingNodeResponse,
    "curriculum / harvest_links": HarvestedLinksResponse,
    "intent_and_clarify / constraints": ClarificationConstraintsResponse,
    "v07 / repl_follow_up": ReplFollowUpResponse,
    "vlm / batch": VlmBatchResponseContract,
    "v07 / chunk_extract": ChunkExtractionContract,
    "v07 / concept_graph": ConceptGraphContract,
    "v07 / profile_gaps": ProfileGapMapContract,
    "v07 / tradeoff_matrix": TradeoffMatrixContract,
    "consensus_query_sanitize": AcademicQueryContract,
    "consensus_validate": ValidationResultContract,
    "profile_relevance_gate": ProfileApplicabilityContract,
    "consensus_refinement_sanitize": RefinementSanitizeContract,
    "source_evaluator_lite": SourceEvaluatorLiteContract,
    "curriculum / lite_query_plan": LiteQueryPlanContract,
    "curriculum / lite_academic_query": LiteAcademicQueryContract,
    "curriculum / lite_batch_eval": LiteBatchEvalContract,
    "curriculum / lite_source_batch": LiteSourceBatchContract,
    "curriculum / lite_site_suggestions": LiteSiteSuggestionsContract,
    "domain_profiler_batch": DomainProfilerBatchContract,
    "exa / search_context_expand": ExaSearchContextExpansion,
    "exa / domain_authority": DomainAuthorityVerdict,
    "exa / domain_authority_batch": BatchDomainAuthorityResponse,
    "ingest_gate / paper_structure": PaperStructureAnalysis,
    "ingest_gate / paper_credibility": PaperCredibilityAnalysis,
    "ingest / tiered_code_prune": TieredClassificationResult,
    "pre_map_dedup / bulk_gate": CanonicalMapContract,
}

__all__ = [
    "GEMINI_STRUCTURED_CONTRACTS",
    "STRUCTURED_LECTURE_FIELD_RULES",
    "ActiveDrillStepResponse",
    "AnswerAccuracyGrade",
    "LayerCompletionTutorOutput",
    "StandardDrillTutorOutput",
    "DrillStepResponse",
    "TechnicalConceptAudit",
    "UnravelingNodeResponse",
    "HarvestedLinksResponse",
    "ClarificationConstraintsResponse",
    "ReplFollowUpResponse",
    "StructuredLectureResponse",
    "structured_lecture_to_dense",
    "IntroAssessmentContract",
    "DeepDiveTutorContract",
    "DeepDiveExplainContract",
    "DeepDiveDeepAnalysisContract",
    "StepAnalysisContract",
    "DialogueFactManifestContract",
    "NodeExplainContract",
    "SubConceptGapEvalContract",
    "EvaluatorCritiqueContract",
    "CurriculumReasonerContract",
    "FlashCurriculumPayloadContract",
    "ModelFirstPayloadContract",
    "GeminiSourcesEnrichmentContract",
    "ExpansionVectorContract",
    "FlashExpansionPatchContract",
    "FinalResponseContract",
    "GeminiL0DecompositionContract",
    "L2EvidenceExtractionContract",
    "ResearchEvaluationContract",
    "AnalysisReportContract",
    "VlmBatchResponseContract",
    "ChunkExtractionContract",
    "ConceptGraphContract",
    "ProfileGapMapContract",
    "TradeoffMatrixContract",
    "AcademicQueryContract",
    "ValidationResultContract",
    "PaperCredibilityAnalysis",
    "PaperStructureAnalysis",
    "TieredClassificationResult",
    "ProfileApplicabilityContract",
    "RefinementSanitizeContract",
    "SourceEvaluatorLiteContract",
    "LiteQueryPlanContract",
    "LiteAcademicQueryContract",
    "ArxivQueryParamsContract",
    "LiteBatchEvalContract",
    "LiteSiteSuggestionsContract",
    "DomainProfilerBatchContract",
    "ExaSearchContextExpansion",
    "DomainAuthorityVerdict",
    "DomainAuthorityItem",
    "BatchDomainAuthorityResponse",
    "CanonicalMapContract",
]
