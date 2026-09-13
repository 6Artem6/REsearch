"""Pre-MAP source deduplication (before the expensive Gemma MAP+REDUCE phase).

See pre_map_deduplicator.py for the entry point
(deduplicate_before_map_reduce). Distinct from services/deduplication/
(entity_consensus_engine.py), which dedupes extracted FACTS/atoms inside one
REDUCE batch — this package dedupes whole SOURCE candidates before MAP even
starts.
"""
