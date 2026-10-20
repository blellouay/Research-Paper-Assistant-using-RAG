import json
import unittest
from types import SimpleNamespace

from src.generation.generator import (
    INSUFFICIENT_EVIDENCE,
    StructuredOutputError,
    _parse_json_object,
    build_final_fact_set,
    extract_claims,
    find_direct_support,
    generate_grounded_answer,
    normalize_supported,
    normalize_status,
    normalize_source_id,
    normalize_sources,
    synthesize_verified_answer,
    verify_claim,
    verify_claims,
    verify_claims_with_evidence,
)
from src.retrieval.claim_evidence import (
    BERT_PAPER,
    TRANSFORMER_PAPER,
    classify_claim_relation,
    resolve_claim_entity,
    resolve_paper_hint,
    retrieve_evidence_for_claim,
)


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content=next(self.responses))
        )


def result(paper, text, section="Architecture"):
    return {
        "payload": {
            "paper_title": paper,
            "section": section,
            "text": text,
        }
    }


class ClaimGroundingTests(unittest.TestCase):
    def test_transformer_reference_is_not_a_second_asserted_subject(self):
        text = "BERT is based on the original Transformer architecture."
        classification = classify_claim_relation(text)
        self.assertEqual(classification["primary_entity"], "BERT")
        self.assertEqual(classification["referenced_entities"], ["Transformer"])
        self.assertEqual(classification["asserted_entities"], ["BERT"])
        self.assertEqual(classification["relation_type"], "REFERENCE_ONLY")
        self.assertEqual(resolve_paper_hint(text), BERT_PAPER)

        evidence = [result(
            BERT_PAPER,
            "BERT is based on the original Transformer architecture.",
            "Model Architecture",
        )]
        verdict = find_direct_support(
            {"id": "claim_1", "text": text, "paper_hint": BERT_PAPER},
            evidence,
            ["Source 1"],
        )
        self.assertEqual(verdict["status"], "SUPPORTED")

    def test_true_comparison_requires_both_asserted_entities(self):
        classification = classify_claim_relation(
            "BERT uses bidirectional attention whereas Transformer uses constrained attention."
        )
        self.assertEqual(classification["relation_type"], "COMPARATIVE")
        self.assertEqual(classification["asserted_entities"], ["BERT", "Transformer"])
        self.assertIsNone(resolve_paper_hint(
            "BERT uses bidirectional attention whereas Transformer uses constrained attention."
        ))

    def test_unsupported_qualifier_is_reduced_to_supported_core(self):
        claim = {
            "id": "claim_1",
            "text": "The Transformer has a more traditional encoder-decoder structure.",
        }
        context = (
            "SOURCE 1\nPaper: Attention Is All You Need\nSection: Model Architecture\n\n"
            "The Transformer follows an encoder-decoder architecture."
        )
        client = ScriptedClient([
            json.dumps({"claim_id": "claim_1", "status": "UNSUPPORTED", "sources": [], "reason": "Traditional is not established."}),
            json.dumps({"claim_id": "claim_1", "status": "SUPPORTED", "sources": ["Source 1"], "reason": "The encoder-decoder structure is explicit."}),
        ])
        verdict = verify_claim(claim, context, client=client)
        self.assertEqual(verdict["status"], "PARTIALLY_SUPPORTED")
        self.assertEqual(
            verdict["supported_part"],
            "The Transformer has an encoder-decoder structure.",
        )
        self.assertEqual(verdict["unsupported_part"], "The characterization 'more traditional'.")

    def test_configuration_specific_transformer_result_becomes_supported_core(self):
        claim = {
            "id": "claim_1",
            "text": "The original Transformer achieved state-of-the-art results on WMT 2014 translation.",
        }
        context = (
            "SOURCE 1\nPaper: Attention Is All You Need\nSection: Machine Translation\n\n"
            "The Transformer (big) achieved a new state of the art on WMT 2014 "
            "English-to-German and English-to-French translation tasks."
        )
        client = ScriptedClient([
            json.dumps({"claim_id": "claim_1", "status": "UNSUPPORTED", "sources": [], "reason": "The configuration is too broad."}),
            json.dumps({"claim_id": "claim_1", "status": "SUPPORTED", "sources": ["Source 1"], "reason": "The big configuration is explicit."}),
        ])
        verdict = verify_claim(claim, context, client=client)
        self.assertEqual(verdict["status"], "PARTIALLY_SUPPORTED")
        self.assertIn("Transformer (big)", verdict["supported_part"])
        self.assertIn("English-to-German", verdict["supported_part"])
        self.assertIn("English-to-French", verdict["supported_part"])

    def test_bert_transformer_encoder_has_bert_as_primary_subject(self):
        classification = classify_claim_relation(
            "BERT is a multi-layer bidirectional Transformer encoder."
        )
        self.assertEqual(classification["primary_entity"], "BERT")
        self.assertEqual(classification["asserted_entities"], ["BERT"])
        self.assertEqual(classification["relation_type"], "SINGLE_SUBJECT")
        self.assertEqual(resolve_paper_hint(
            "BERT is a multi-layer bidirectional Transformer encoder."
        ), BERT_PAPER)

    def test_context_heading_resolves_implicit_bert_subject(self):
        draft = "BERT\n- The model architecture is a multi-layer bidirectional Transformer encoder."
        client = ScriptedClient([json.dumps({"claims": [{
            "id": "claim_1",
            "text": "The model architecture is a multi-layer bidirectional Transformer encoder.",
        }]})])
        claims = extract_claims(draft, client=client)
        self.assertEqual(
            claims[0]["text"],
            "BERT's model architecture is a multi-layer bidirectional Transformer encoder.",
        )
        self.assertEqual(claims[0]["entities"], ["BERT"])
        self.assertEqual(claims[0]["paper_hint"], BERT_PAPER)
        self.assertEqual(claims[0]["original_draft_section"], "BERT")

    def test_source_collection_normalization_accepts_safe_variants(self):
        valid = {"Source 6"}
        variants = (
            "Source 6", ["Source 6"], "[Source 6]", 6, [6],
            {"source": "Source 6"}, [{"source": "Source 6"}],
        )
        for value in variants:
            with self.subTest(value=value):
                self.assertEqual(normalize_sources(value, valid), ["Source 6"])
        self.assertEqual(normalize_sources("Source 99", valid), [])

    def test_related_sentence_task_evidence_does_not_support_wordpiece(self):
        claim = {
            "id": "claim_1",
            "text": "BERT uses WordPiece embeddings.",
            "entities": ["BERT"],
            "paper_hint": BERT_PAPER,
        }
        package = {
            "context": "SOURCE 1\nPaper: BERT\nSection: Results\n\nBERT achieves strong results on sentence-level tasks.",
            "results": [result("BERT", "BERT achieves strong results on sentence-level tasks.", "Results")],
            "source_ids": ["Source 1"],
            "evidence_used": [{"source": "Source 1", "paper": "BERT", "section": "Results"}],
        }
        client = ScriptedClient([json.dumps({
            "claim_id": "claim_1", "status": "SUPPORTED",
            "sources": {"source": "Source 1"},
            "reason": "The excerpt discusses BERT sentence tasks.",
        })])
        verdicts, _ = verify_claims_with_evidence(
            [claim], {"claim_1": package}, client=client, use_direct_match=False
        )
        self.assertEqual(verdicts[0]["status"], "UNSUPPORTED")
        self.assertEqual(verdicts[0]["method"], "technical_evidence_guard")

    def test_valid_partial_comparison_keeps_only_supported_transformer_side(self):
        claim = {
            "id": "claim_1",
            "text": "Both BERT and Transformer use multi-headed self-attention.",
            "entities": ["BERT", "Transformer"],
            "paper_hint": None,
        }
        package = {
            "context": "SOURCE 1\nPaper: Attention Is All You Need\nSection: Model Architecture\n\nThe original Transformer uses multi-headed self-attention.",
            "results": [result(TRANSFORMER_PAPER, "The original Transformer uses multi-headed self-attention.", "Model Architecture")],
            "source_ids": ["Source 1"],
            "evidence_used": [{"source": "Source 1", "paper": TRANSFORMER_PAPER, "section": "Model Architecture"}],
        }
        client = ScriptedClient([json.dumps({
            "claim_id": "claim_1", "status": "PARTIALLY_SUPPORTED",
            "sources": ["Source 1"],
            "supported_part": "The original Transformer uses multi-headed self-attention.",
            "unsupported_part": "The evidence does not establish that BERT uses multi-headed self-attention.",
            "reason": "Only the Transformer side is stated.",
        })])
        verdicts, _ = verify_claims_with_evidence(
            [claim], {"claim_1": package}, client=client, use_direct_match=False
        )
        facts = build_final_fact_set([], verdicts)
        self.assertEqual(verdicts[0]["status"], "PARTIALLY_SUPPORTED")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["entities"], ["Transformer"])
        self.assertNotIn("Both BERT", facts[0]["text"])

    def test_multi_entity_claim_cannot_be_supported_by_bert_only(self):
        claim = {
            "id": "claim_1",
            "text": "Both BERT and Transformer achieved state-of-the-art results.",
            "entities": ["BERT", "Transformer"],
            "paper_hint": None,
        }
        package = {
            "context": "SOURCE 1\nPaper: BERT\nSection: Results\n\nBERT achieved state-of-the-art results.",
            "results": [result("BERT", "BERT achieved state-of-the-art results.", "Results")],
            "source_ids": ["Source 1"],
            "evidence_used": [{"source": "Source 1", "paper": "BERT", "section": "Results"}],
        }
        client = ScriptedClient([json.dumps({
            "claim_id": "claim_1", "status": "SUPPORTED",
            "sources": ["Source 1"], "reason": "BERT evidence states results.",
        })])
        verdicts, _ = verify_claims_with_evidence(
            [claim], {"claim_1": package}, client=client, use_direct_match=False
        )
        self.assertEqual(verdicts[0]["status"], "UNSUPPORTED")
        self.assertEqual(verdicts[0]["method"], "multi_entity_guard")

    def test_malformed_partial_comparison_is_error_and_never_final_fact(self):
        claim = {"id": "claim_1", "text": "Both BERT and Transformer use attention."}
        malformed = json.dumps({
            "claim_id": "claim_1", "status": "PARTIALLY_SUPPORTED",
            "sources": ["Source 1"],
            "supported_part": claim["text"], "unsupported_part": claim["text"],
            "reason": "Malformed partial.",
        })
        context = "SOURCE 1\nPaper: Paper\nSection: S\n\nTransformer uses attention."
        verdicts = verify_claims(
            [claim], context, client=ScriptedClient([malformed, malformed])
        )
        self.assertTrue(verdicts[0]["verification_error"])
        self.assertEqual(build_final_fact_set([], verdicts), [])

    def test_hidden_size_claim_splits_by_bert_variant(self):
        client = ScriptedClient([json.dumps({"claims": [{
            "id": "claim_1", "text": "BERT has a hidden size of 768 and 1024."
        }]})])
        claims = extract_claims("draft", client=client)
        self.assertEqual(
            [claim["text"] for claim in claims],
            ["BERT BASE has hidden size 768.", "BERT LARGE has hidden size 1024."],
        )
        self.assertEqual(
            [claim["entities"] for claim in claims],
            [["BERT BASE"], ["BERT LARGE"]],
        )

    def test_transformer_translation_result_keeps_wmt_scope_from_bullet(self):
        draft = (
            "Original Transformer Architecture\n"
            "- The original Transformer achieves a new state of the art on "
            "translation tasks, such as WMT 2014 English-to-German and "
            "English-to-French translation tasks."
        )
        client = ScriptedClient([json.dumps({"claims": [{
            "id": "claim_1",
            "text": "The original Transformer achieves a new state of the art on translation tasks.",
        }]})])
        claims = extract_claims(draft, client=client)
        self.assertEqual(
            claims[0]["text"],
            "The original Transformer achieved state-of-the-art results on the reported WMT 2014 English-to-German and English-to-French translation tasks.",
        )
        self.assertEqual(claims[0]["paper_hint"], TRANSFORMER_PAPER)

    def test_entity_to_paper_resolution_keeps_models_distinct(self):
        cases = [
            ("BERT uses bidirectional self-attention.", "BERT", BERT_PAPER),
            ("The original Transformer uses an encoder-decoder architecture.", "Transformer", TRANSFORMER_PAPER),
            ("The GPT Transformer uses constrained self-attention.", "GPT", None),
            ("BERT LARGE contains 340M parameters.", "BERT LARGE", BERT_PAPER),
        ]
        for text, entity, paper in cases:
            with self.subTest(text=text):
                self.assertEqual(resolve_claim_entity(text), entity)
                self.assertEqual(resolve_paper_hint(text), paper)

    def test_claim_retrieval_uses_strict_target_paper(self):
        class Store:
            def __init__(self):
                self.calls = []

            def search_by_paper(self, query_vector, paper_title, limit):
                self.calls.append(paper_title)
                return [SimpleNamespace(
                    payload={"paper_title": paper_title, "text": "evidence"},
                    score=1.0,
                )]

            def search(self, query_vector, limit):
                raise AssertionError("global search must not be used")

        class Embedder:
            def embed_query(self, query):
                return [0.1]

        class Reranker:
            def rerank(self, query, results, top_k):
                return [{"payload": item.payload, "score": 1.0} for item in results]

        store = Store()
        for claim in (
            "The original Transformer has an encoder-decoder structure.",
            "The original Transformer is trained significantly faster than recurrent architectures.",
        ):
            evidence = retrieve_evidence_for_claim(
                claim, store, Embedder(), Reranker(),
                paper_hint=resolve_paper_hint(claim),
            )
            self.assertEqual(evidence[0]["payload"]["paper_title"], TRANSFORMER_PAPER)
        self.assertEqual(store.calls, [TRANSFORMER_PAPER, TRANSFORMER_PAPER])

    def test_mismatched_paper_evidence_becomes_routing_error(self):
        claim = {
            "id": "claim_1",
            "text": "The original Transformer has an encoder-decoder structure.",
            "entities": ["Transformer"],
            "paper_hint": TRANSFORMER_PAPER,
        }
        package = {
            "context": "SOURCE 1\nPaper: BERT\nSection: BERT\n\nBERT uses bidirectional attention.",
            "results": [result("BERT", "BERT uses bidirectional attention.")],
            "source_ids": ["Source 1"],
            "evidence_used": [{"source": "Source 1", "paper": "BERT", "section": "BERT"}],
        }
        client = ScriptedClient([])
        verdicts, _ = verify_claims_with_evidence(
            [claim], {"claim_1": package}, client=client, use_direct_match=False
        )
        self.assertTrue(verdicts[0]["verification_error"])
        self.assertEqual(verdicts[0]["method"], "evidence_scope_guard")
        self.assertIn("target paper", verdicts[0]["reason"])
        self.assertEqual(client.calls, [])

    def test_supported_verdict_citing_wrong_paper_fails_final_invariant(self):
        claim = {
            "id": "claim_1",
            "text": "The original Transformer has an encoder-decoder structure.",
            "entities": ["Transformer"],
            "paper_hint": TRANSFORMER_PAPER,
        }
        package = {
            "context": (
                "SOURCE 1\nPaper: Attention Is All You Need\nSection: Model Architecture\n\nEncoder-decoder.\n\n"
                "SOURCE 2\nPaper: BERT\nSection: BERT\n\nBidirectional attention."
            ),
            "results": [
                result(TRANSFORMER_PAPER, "Encoder-decoder.", "Model Architecture"),
                result("BERT", "Bidirectional attention.", "BERT"),
            ],
            "source_ids": ["Source 1", "Source 2"],
            "evidence_used": [
                {"source": "Source 1", "paper": TRANSFORMER_PAPER, "section": "Model Architecture"},
                {"source": "Source 2", "paper": "BERT", "section": "BERT"},
            ],
        }
        client = ScriptedClient([json.dumps({
            "claim_id": "claim_1",
            "status": "SUPPORTED",
            "sources": ["Source 2"],
            "reason": "Source 2 supports the claim.",
        })])
        verdicts, _ = verify_claims_with_evidence(
            [claim], {"claim_1": package}, client=client, use_direct_match=False
        )
        self.assertTrue(verdicts[0]["verification_error"])
        self.assertEqual(verdicts[0]["status"], "UNSUPPORTED")
        self.assertEqual(verdicts[0]["sources"], [])

    def test_bert_stage_wording_is_normalized_during_extraction(self):
        client = ScriptedClient([json.dumps({"claims": [{
            "id": "claim_1",
            "text": "BERT has two pre-trained tasks: pre-training and fine-tuning.",
            "entities": ["BERT"],
        }]})])
        claims = extract_claims("draft", client=client)
        self.assertEqual(
            claims[0]["text"],
            "BERT's framework has two stages: pre-training and fine-tuning.",
        )
        self.assertEqual(claims[0]["paper_hint"], BERT_PAPER)

    def test_gpt_claim_can_use_explicit_bert_passage_without_reclassification(self):
        claim = {
            "id": "claim_gpt",
            "text": "The GPT Transformer uses constrained self-attention.",
            "entities": ["GPT"],
            "paper_hint": None,
        }
        evidence = [result(
            BERT_PAPER,
            "The GPT Transformer uses constrained self-attention.",
            section="BERT",
        )]
        verdict = find_direct_support(claim, evidence, ["Source 1"])
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["status"], "SUPPORTED")
        self.assertEqual(resolve_claim_entity(claim["text"]), "GPT")
        self.assertIsNone(resolve_paper_hint(claim["text"]))

    def test_bert_transformer_rejects_unsupported_architecture_claims(self):
        retrieved = [
            result(
                "BERT",
                "BERT is pre-trained on unlabeled data. Its Transformer encoder "
                "uses bidirectional self-attention.",
            ),
            result(
                "Attention Is All You Need",
                "The Transformer follows an encoder-decoder architecture. The "
                "encoder contains self-attention and feed-forward sub-layers.",
            ),
        ]
        claims = [
            {"id": "claim_1", "text": "BERT is pre-trained on unlabeled data."},
            {"id": "claim_2", "text": "The original Transformer uses unidirectional self-attention."},
            {"id": "claim_3", "text": "The original Transformer is trained directly on labeled data."},
            {"id": "claim_4", "text": "Both BERT and the original Transformer use a bidirectional encoder and decoder."},
        ]
        responses = [
            "A deliberately imperfect draft.",
            json.dumps({"claims": claims}),
            json.dumps({"claim_id": "claim_1", "supported": True, "sources": ["Source 1"], "reason": "Explicitly stated."}),
            json.dumps({"claim_id": "claim_2", "supported": False, "sources": [], "reason": "No excerpt states this."}),
            json.dumps({"claim_id": "claim_3", "supported": False, "sources": [], "reason": "Training supervision is not stated."}),
            json.dumps({"claim_id": "claim_4", "supported": False, "sources": [], "reason": "Neither side of the full comparison is supported."}),
            json.dumps({"claim_id": "claim_5", "supported": False, "sources": [], "reason": "The Transformer evidence does not establish a bidirectional encoder and decoder."}),
        ]

        output = generate_grounded_answer(
            "Compare BERT with the original Transformer architecture.",
            retrieved,
            top_k=2,
            comparison=True,
            client=ScriptedClient(responses),
            use_direct_match=False,
        )

        self.assertEqual(len(output["verified_claims"]), 5)
        self.assertEqual(len(output["supported_claims"]), 1)
        self.assertEqual(len(output["rejected_claims"]), 4)
        self.assertIn("BERT is pre-trained on unlabeled data", output["final_answer"])
        self.assertIn("[Source 1]", output["final_answer"])
        self.assertNotIn("unidirectional", output["final_answer"])
        self.assertNotIn("trained directly on labeled data", output["final_answer"])
        self.assertNotIn("bidirectional encoder and decoder", output["final_answer"])

    def test_simple_encoder_sublayers_keeps_supported_claims(self):
        retrieved = [
            result(
                "Attention Is All You Need",
                "Each encoder layer has two sub-layers. The first is a multi-head "
                "self-attention mechanism, and the second is a position-wise fully "
                "connected feed-forward network.",
            )
        ]
        claims = [
            {"id": "claim_1", "text": "The first encoder sub-layer is multi-head self-attention."},
            {"id": "claim_2", "text": "The second encoder sub-layer is a position-wise feed-forward network."},
        ]
        responses = [
            "The layer has attention and feed-forward sub-layers.",
            json.dumps({"claims": claims}),
            json.dumps({"claim_id": "claim_1", "supported": True, "sources": ["Source 1"], "reason": "Explicitly stated."}),
            json.dumps({"claim_id": "claim_2", "supported": True, "sources": ["Source 1"], "reason": "Explicitly stated."}),
        ]

        output = generate_grounded_answer(
            "What are the two sub-layers in each Transformer encoder layer?",
            retrieved,
            client=ScriptedClient(responses),
            use_direct_match=False,
        )

        self.assertEqual(len(output["verified_claims"]), 2)
        self.assertEqual(len(output["supported_claims"]), 2)
        self.assertEqual(output["rejected_claims"], [])
        self.assertEqual(output["final_answer"].count("[Source 1]"), 2)

    def test_insufficient_evidence_rejects_all_claims(self):
        retrieved = [result("Paper", "This excerpt discusses an unrelated topic.")]
        responses = [
            "The system uses ten layers.",
            json.dumps({"claims": [{"id": "claim_1", "text": "The system uses ten layers."}]}),
            json.dumps({"claim_id": "claim_1", "supported": False, "sources": [], "reason": "The excerpt gives no layer count."}),
        ]

        output = generate_grounded_answer(
            "How many layers does the system use?",
            retrieved,
            client=ScriptedClient(responses),
            use_direct_match=False,
        )

        self.assertEqual(len(output["verified_claims"]), 1)
        self.assertEqual(output["supported_claims"], [])
        self.assertEqual(len(output["rejected_claims"]), 1)
        self.assertEqual(output["final_answer"], INSUFFICIENT_EVIDENCE)

    def test_json_parser_handles_fences_and_rejects_malformed_output(self):
        parsed = _parse_json_object('```json\n{"claims": []}\n```')
        self.assertEqual(parsed, {"claims": []})
        with self.assertRaises(StructuredOutputError):
            _parse_json_object("not JSON")
        with self.assertRaises(StructuredOutputError):
            _parse_json_object(
                '{"claims": [{"id": "claim_1"}], '
                '"claims": [{"id": "claim_12"}]}'
            )

    def test_verifier_uses_caller_claim_id_when_model_renumbers_it(self):
        client = ScriptedClient([
            json.dumps({
                "claim_id": "claim_1",
                "supported": True,
                "sources": ["Source 1"],
                "reason": "The source explicitly supports the claim.",
            })
        ])
        claim = {"id": "claim_7", "text": "The encoder uses self-attention."}
        context = "SOURCE 1\nPaper: Paper\nSection: Architecture\n\nThe encoder uses self-attention."

        verdict = verify_claim(claim, context, client=client)

        self.assertEqual(verdict["claim_id"], "claim_7")
        self.assertTrue(verdict["supported"])

    def test_supported_field_normalization(self):
        for value in (True, "true", "SUPPORTED", "yes"):
            self.assertTrue(normalize_supported(value))
        for value in (False, "false", "UNSUPPORTED", "no"):
            self.assertFalse(normalize_supported(value))
        with self.assertRaises(ValueError):
            normalize_supported("maybe")

    def test_verification_error_is_preserved_and_counted(self):
        claims = [
            {"id": "claim_1", "text": "Supported fact."},
            {"id": "claim_2", "text": "Claim with malformed verdict."},
        ]
        client = ScriptedClient([
            json.dumps({
                "claim_id": "claim_1",
                "supported": "SUPPORTED",
                "sources": ["Source 1"],
                "reason": "Source 1 explicitly states this fact.",
            }),
            json.dumps({
                "claim_id": "claim_2",
                "supported": "maybe",
                "sources": [],
                "reason": "Invalid verdict value.",
            }),
        ])
        context = "SOURCE 1\nPaper: Paper\nSection: S\n\nSupported fact."

        results = verify_claims(claims, context, client=client)

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["supported"])
        self.assertTrue(results[1]["verification_error"])
        self.assertIn("invalid verification status", results[1]["reason"])

    def test_semantic_entailment_statuses_and_partial_claim(self):
        context = (
            "SOURCE 1\nPaper: BERT\nSection: Architecture\n\n"
            "BERT's model architecture is a multi-layer bidirectional Transformer encoder. "
            "During pre-training, the model is trained on unlabeled data. "
            "The input representation can unambiguously represent both a single sentence "
            "and a sequence of sentences (e.g., Question, Answer) in one token sequence.\n\n"
            "SOURCE 2\nPaper: Transformer\nSection: Architecture\n\n"
            "The Transformer uses an encoder-decoder architecture."
        )
        responses = [
            json.dumps({"claim_id": "a", "status": "SUPPORTED", "sources": ["Source 1"], "reason": "Multi-layer semantically entails deep in this architectural description."}),
            json.dumps({"claim_id": "b", "status": "SUPPORTED", "sources": ["Source 1"], "reason": "The source explicitly states pre-training uses unlabeled data."}),
            json.dumps({"claim_id": "c", "status": "SUPPORTED", "sources": ["Source 1"], "reason": "The sentence-sequence example semantically supports sentence pairs."}),
            json.dumps({"claim_id": "d", "status": "UNSUPPORTED", "sources": [], "reason": "The excerpts describe architecture but never state that self-attention is unidirectional."}),
            json.dumps({"claim_id": "e", "status": "PARTIALLY_SUPPORTED", "sources": ["Source 1"], "supported_part": "BERT is pretrained on unlabeled data.", "unsupported_part": "The Transformer is trained only on labeled data.", "reason": "Source 1 supports BERT pre-training, but no excerpt states that the Transformer is trained only on labeled data."}),
        ]
        client = ScriptedClient(responses)
        claims = [
            {"id": "a", "text": "BERT is a deep bidirectional Transformer encoder."},
            {"id": "b", "text": "BERT is pre-trained on unlabeled data."},
            {"id": "c", "text": "BERT can represent single sentences and sentence pairs in one token sequence."},
            {"id": "d", "text": "The original Transformer uses unidirectional self-attention."},
            {"id": "e", "text": "BERT is pretrained on unlabeled data and the Transformer is trained only on labeled data."},
        ]

        verdicts = [verify_claim(claim, context, client=client) for claim in claims]

        self.assertEqual(
            [verdict["status"] for verdict in verdicts],
            ["SUPPORTED", "SUPPORTED", "SUPPORTED", "UNSUPPORTED", "PARTIALLY_SUPPORTED"],
        )
        self.assertEqual(
            verdicts[4]["supported_part"],
            "BERT is pretrained on unlabeled data.",
        )
        self.assertIn("semantic entailment", client.calls[0]["messages"][0]["content"])

    def test_status_normalization(self):
        self.assertEqual(normalize_status("supported"), "SUPPORTED")
        self.assertEqual(normalize_status("partially supported"), "PARTIALLY_SUPPORTED")
        self.assertEqual(normalize_status("unsupported"), "UNSUPPORTED")

    def test_exact_semantic_failures_have_expected_statuses(self):
        context = (
            "SOURCE 1\nPaper: BERT\nSection: Architecture\n\n"
            "BERT's model architecture is a multi-layer bidirectional Transformer "
            "encoder based on the original implementation described in Vaswani et al. (2017). "
            "BERT uses bidirectional self-attention.\n\n"
            "SOURCE 2\nPaper: Transformer\nSection: Results\n\n"
            "The Transformer can be trained significantly faster than architectures "
            "based on recurrent or convolutional layers.\n\n"
            "SOURCE 3\nPaper: Transformer\nSection: Architecture\n\n"
            "The encoder maps input representations to continuous representations. "
            "The decoder generates output symbols one element at a time.\n\n"
            "SOURCE 4\nPaper: Transformer\nSection: Decoder\n\n"
            "The decoder prevents positions from attending to subsequent positions."
        )
        claims = [
            {"id": "claim_1", "text": "BERT is a deep bidirectional Transformer encoder based on the original Transformer architecture."},
            {"id": "claim_2", "text": "The Transformer is trained significantly faster than architectures based on recurrent or convolutional layers."},
            {"id": "claim_3", "text": "The Transformer has an encoder-decoder architecture."},
            {"id": "claim_4", "text": "BERT uses bidirectional self-attention."},
            {"id": "claim_5", "text": "The original Transformer uses constrained self-attention."},
            {"id": "claim_6", "text": "The Transformer is trained only on labeled data."},
        ]
        responses = [
            json.dumps({"claim_id": "claim_1", "status": "SUPPORTED", "sources": ["Source 1"], "reason": "Source 1 describes a multi-layer bidirectional encoder based on Vaswani et al., semantically supporting the claim."}),
            json.dumps({"claim_id": "claim_2", "status": "SUPPORTED", "sources": ["Source 2"], "reason": "Source 2 directly states the faster-training comparison."}),
            json.dumps({"claim_id": "claim_3", "status": "SUPPORTED", "sources": ["Source 3"], "reason": "Source 3 describes complementary encoder and decoder behavior, establishing the architecture."}),
            json.dumps({"claim_id": "claim_4", "status": "SUPPORTED", "sources": ["Source 1"], "reason": "Source 1 explicitly states that BERT uses bidirectional self-attention."}),
            json.dumps({"claim_id": "claim_5", "status": "UNSUPPORTED", "sources": [], "reason": "Source 4 discusses masking only in the decoder and does not support generalizing constrained attention to the entire Transformer."}),
            json.dumps({"claim_id": "claim_6", "status": "UNSUPPORTED", "sources": [], "reason": "No excerpt describes the Transformer's training data or states that it uses only labeled data."}),
        ]

        verdicts = verify_claims(claims, context, client=ScriptedClient(responses))

        self.assertEqual(
            [item["status"] for item in verdicts],
            ["SUPPORTED", "SUPPORTED", "SUPPORTED", "SUPPORTED", "UNSUPPORTED", "UNSUPPORTED"],
        )

    def test_inconsistent_unsupported_reason_triggers_retry(self):
        context = (
            "SOURCE 1\nPaper: Transformer\nSection: Results\n\n"
            "The Transformer can be trained significantly faster than recurrent models."
        )
        client = ScriptedClient([
            json.dumps({"claim_id": "claim_1", "status": "UNSUPPORTED", "sources": [], "reason": "Source 1 explicitly states that the Transformer can be trained significantly faster."}),
            json.dumps({"claim_id": "claim_1", "status": "SUPPORTED", "sources": ["Source 1"], "reason": "Source 1 directly states the claim."}),
        ])

        verdict = verify_claim(
            {"id": "claim_1", "text": "The Transformer is trained significantly faster than recurrent models."},
            context,
            client=client,
        )

        self.assertEqual(verdict["status"], "SUPPORTED")
        self.assertEqual(len(client.calls), 2)

    def test_partial_missing_optional_unsupported_part_is_normalized(self):
        context = "SOURCE 1\nPaper: BERT\nSection: Training\n\nBERT is pretrained on unlabeled data."
        client = ScriptedClient([
            json.dumps({
                "claim_id": "claim_1",
                "status": "PARTIALLY_SUPPORTED",
                "sources": ["Source 1"],
                "supported_part": "BERT is pretrained on unlabeled data.",
                "reason": "Only the BERT portion is supported.",
            })
        ])

        verdict = verify_claim(
            {"id": "claim_1", "text": "BERT is pretrained on unlabeled data and is more expensive."},
            context,
            client=client,
        )

        self.assertEqual(verdict["status"], "PARTIALLY_SUPPORTED")
        self.assertEqual(
            verdict["unsupported_part"],
            "BERT is pretrained on unlabeled data and is more expensive.",
        )

    def test_comparative_claim_is_split_into_atomic_claims(self):
        client = ScriptedClient([
            json.dumps({
                "claims": [{
                    "id": "claim_1",
                    "text": "BERT uses bidirectional self-attention, whereas the original Transformer uses constrained self-attention.",
                    "entities": ["BERT", "Transformer"],
                }]
            })
        ])

        claims = extract_claims("draft", client=client)

        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["text"], "BERT uses bidirectional self-attention")
        self.assertEqual(
            claims[1]["text"],
            "the original Transformer uses constrained self-attention",
        )
        self.assertEqual(claims[0]["entities"], ["BERT"])
        self.assertEqual(claims[1]["entities"], ["Transformer"])

    def test_source_id_normalization_accepts_common_variations(self):
        available = {"Source 15", "Source 7"}
        for value in ("Source 15", "source 15", "[Source 15]", "15", "Evidence 15"):
            self.assertEqual(normalize_source_id(value, available), "Source 15")
        self.assertIsNone(normalize_source_id("Source 4", available))

        context = "SOURCE 15\nPaper: Transformer\nSection: Architecture\n\nEncoder-decoder."
        client = ScriptedClient([json.dumps({
            "claim_id": "claim_1",
            "status": "SUPPORTED",
            "sources": ["15"],
            "reason": "Evidence 15 explicitly states the architecture.",
        })])
        verdict = verify_claim(
            {"id": "claim_1", "text": "The Transformer has an encoder-decoder architecture."},
            context,
            client=client,
        )
        self.assertEqual(verdict["sources"], ["Source 15"])
        self.assertEqual(len(client.calls), 1)

    def test_invalid_supported_source_is_retried_once(self):
        context = "SOURCE 15\nPaper: Transformer\nSection: Architecture\n\nEncoder-decoder."
        client = ScriptedClient([
            json.dumps({"claim_id": "claim_1", "status": "SUPPORTED", "sources": ["Source 99"], "reason": "The evidence states it."}),
            json.dumps({"claim_id": "claim_1", "status": "SUPPORTED", "sources": ["Source 15"], "reason": "Source 15 states it."}),
        ])
        verdict = verify_claim(
            {"id": "claim_1", "text": "The Transformer has an encoder-decoder architecture."},
            context,
            client=client,
        )
        self.assertEqual(verdict["sources"], ["Source 15"])
        self.assertEqual(len(client.calls), 2)

    def test_final_fact_set_deduplicates_exact_and_overlapping_facts(self):
        supported = [
            {"status": "SUPPORTED", "text": "BERT uses bidirectional self-attention.", "sources": ["Source 1"], "entities": ["BERT"]},
            {"status": "SUPPORTED", "text": "BERT uses bidirectional self-attention", "sources": ["Source 6"], "entities": ["BERT"]},
            {"status": "SUPPORTED", "text": "BERT has a unified architecture across tasks.", "sources": ["Source 1"], "entities": ["BERT"]},
            {"status": "SUPPORTED", "text": "BERT has a unified architecture across tasks, with minimal differences between pretraining and downstream architectures.", "sources": ["Source 2"], "entities": ["BERT"]},
        ]
        facts = build_final_fact_set(supported, [])
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0]["sources"], ["Source 1", "Source 6"])
        self.assertIn("minimal differences", facts[1]["text"])
        self.assertEqual(facts[1]["sources"], ["Source 1", "Source 2"])

    def test_partial_claim_contributes_only_clean_supported_part(self):
        partial = [{
            "status": "PARTIALLY_SUPPORTED",
            "text": "The architecture is identical to Transformer, with a multi-layer bidirectional encoder.",
            "supported_part": "Model Architecture BERT's model architecture is a multi-layer bidirectional Transformer encoder based on...",
            "unsupported_part": "The evidence does not establish that the architecture is identical.",
            "sources": ["Source 1"],
            "entities": ["BERT"],
        }]
        facts = build_final_fact_set([], partial)
        self.assertEqual(len(facts), 1)
        self.assertEqual(
            facts[0]["text"],
            "BERT uses a multi-layer bidirectional Transformer encoder.",
        )
        self.assertNotIn("identical", facts[0]["text"])

    def test_comparison_synthesis_uses_only_normalized_verified_facts(self):
        facts = [
            {"text": "BERT uses a multi-layer bidirectional Transformer encoder.", "sources": ["Source 1"], "entities": ["BERT"], "paper_hint": "BERT"},
            {"text": "The original Transformer has an encoder-decoder architecture.", "sources": ["Source 15"], "entities": ["Transformer"], "paper_hint": "Attention Is All You Need"},
        ]
        answer = synthesize_verified_answer(
            "Compare BERT with the original Transformer architecture.", facts
        )
        self.assertIn("BERT", answer)
        self.assertIn("Original Transformer", answer)
        self.assertIn("encoder-decoder", answer)
        self.assertNotIn("constrained self-attention", answer)
        self.assertNotIn("labeled data", answer)
        self.assertEqual(answer.count("multi-layer bidirectional"), 1)

    def test_direct_support_shortcut_and_entity_filter(self):
        bert_evidence = [result(
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "The BERT Transformer uses bidirectional self-attention.",
        )]
        bert_claim = {
            "id": "claim_1",
            "text": "BERT uses bidirectional self-attention.",
            "paper_hint": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        }
        verdict = find_direct_support(bert_claim, bert_evidence, ["Source 1"])
        self.assertEqual(verdict["status"], "SUPPORTED")
        self.assertEqual(verdict["method"], "direct_match")

        speed_claim = {
            "id": "claim_speed",
            "text": "The Transformer is trained significantly faster than recurrent or convolutional architectures.",
            "paper_hint": "Attention Is All You Need",
        }
        speed_evidence = [result(
            "Attention Is All You Need",
            "The Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.",
        )]
        self.assertEqual(
            find_direct_support(speed_claim, speed_evidence, ["Source 2"])["status"],
            "SUPPORTED",
        )

        contaminated_claim = {
            "id": "claim_2",
            "text": "The original Transformer is pretrained on unlabeled data.",
            "paper_hint": "Attention Is All You Need",
        }
        contaminated_evidence = [result(
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "BERT is pretrained on unlabeled data.",
        )]
        self.assertIsNone(
            find_direct_support(contaminated_claim, contaminated_evidence, ["Source 2"])
        )

    def test_duplicate_claims_reuse_one_verdict(self):
        claims = [
            {"id": "claim_1", "text": "BERT uses bidirectional self-attention.", "entities": ["BERT"], "paper_hint": None},
            {"id": "claim_2", "text": "  bert uses bidirectional self-attention  ", "entities": ["BERT"], "paper_hint": None},
        ]
        package = {
            "context": "SOURCE 1\nPaper: BERT\nSection: A\n\nBERT uses bidirectional self-attention.",
            "results": [result("BERT", "BERT uses bidirectional self-attention.")],
            "source_ids": ["Source 1"],
            "evidence_used": [{"source": "Source 1", "paper": "BERT", "section": "A"}],
        }
        client = ScriptedClient([
            json.dumps({"claim_id": "claim_1", "status": "SUPPORTED", "sources": ["Source 1"], "reason": "Source 1 directly states the claim."})
        ])

        verdicts, cache_hits = verify_claims_with_evidence(
            claims,
            {"claim_1": package, "claim_2": package},
            client=client,
            use_direct_match=False,
        )

        self.assertEqual(cache_hits, 1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(verdicts[0]["status"], verdicts[1]["status"])
        self.assertTrue(verdicts[1]["cache_hit"])


if __name__ == "__main__":
    unittest.main()
