import unittest

from evaluation.evaluation_utils import (
    calculate_fact_coverage,
    expected_paper_coverage,
    hit_at_k,
    is_abstention,
    reciprocal_rank,
    section_matches,
    semantic_match,
    unsupported_claim_leakage,
    validate_citations,
)


class EvaluationUtilityTests(unittest.TestCase):
    def test_section_matching_accepts_parent_child_names(self):
        self.assertTrue(section_matches(
            "Model Architecture > Encoder and Decoder Stacks",
            "Encoder and Decoder Stacks",
        ))
        self.assertTrue(section_matches("multi-head attention", "Multi-Head Attention"))
        self.assertFalse(section_matches("Optimizer", "Positional Encoding"))

    def test_hit_at_k_and_mrr(self):
        relevance = [False, False, True, False]
        self.assertEqual(hit_at_k(relevance, 1), 0)
        self.assertEqual(hit_at_k(relevance, 3), 1)
        self.assertAlmostEqual(reciprocal_rank(relevance), 1 / 3)
        self.assertEqual(reciprocal_rank([False, False]), 0.0)

    def test_expected_paper_coverage(self):
        retrieved = ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"]
        expected = [
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "Attention Is All You Need",
        ]
        self.assertEqual(expected_paper_coverage(retrieved, expected), 0.5)

    def test_citation_validation(self):
        answer = "BERT uses bidirectional attention. [Source 2]"
        facts = [{"text": "BERT uses bidirectional attention.", "sources": ["Source 2"]}]
        valid = validate_citations(answer, facts, {"Source 2"})
        self.assertEqual(valid["citation_validity"], 1.0)
        invalid = validate_citations(answer, facts, {"Source 1"})
        self.assertEqual(invalid["citation_validity"], 0.0)
        self.assertEqual(invalid["invalid_citations"], ["Source 2"])

    def test_unsupported_claim_leakage(self):
        unsupported = [{"text": "The Transformer is trained only on labeled data."}]
        answer = "The Transformer is trained only on labeled data. [Source 1]"
        self.assertEqual(len(unsupported_claim_leakage(answer, unsupported)), 1)
        self.assertEqual(unsupported_claim_leakage("Insufficient evidence.", unsupported), [])

    def test_semantic_expected_fact_matching(self):
        expected = "The original Transformer uses an encoder-decoder architecture."
        generated = "The original Transformer has an encoder-decoder structure."
        self.assertTrue(semantic_match(expected, generated))
        coverage, missing = calculate_fact_coverage(
            [expected], [{"text": generated}]
        )
        self.assertEqual(coverage, 1.0)
        self.assertEqual(missing, [])

    def test_abstention_detection(self):
        self.assertTrue(is_abstention(
            "The retrieved excerpts do not provide enough information to answer this question."
        ))
        self.assertFalse(is_abstention("BERT uses bidirectional self-attention."))


if __name__ == "__main__":
    unittest.main()
