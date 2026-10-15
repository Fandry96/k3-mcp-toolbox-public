#!/usr/bin/env python3
"""
Unit tests for k3_mrl_memory.py:
- _classify_key across all real key shapes and edge cases
- Default exclusions ('tool', 'skill', 'rules' excluded when filter_type=None)
- Explicit filter_type ('tool', 'skill', 'rules', 'brain', 'knowledge', 'research', 'book', 'all')
- mtime formatting and max_age_days filtering
- Contrastive tool docstrings
"""

import datetime
import sys
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

SERVER_DIR = Path(__file__).resolve().parent / "servers"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import k3_mrl_memory as mrl  # noqa: E402


class TestMrlMemory(unittest.TestCase):
    def test_classify_key_all_shapes(self):
        cases = [
            # MCP tools (including tool names containing 'skill' or 'research')
            ("mcp:chrome-devtools-mcp.click::main", "tool"),
            ("mcp:k3-mrl-memory.mrl_search_skills::main", "tool"),
            ("mcp:perplexity-ask.perplexity_ask::main", "tool"),
            # Skills (prefixed and C:/ path skills containing /rules/ or /knowledge/)
            ("skill:code-reviewer::chunk[0]", "skill"),
            ("skill:proto-book/forge-draft::chunk[0]", "skill"),
            ("C:/K3_Firehose/.agent/skills/react-best-practices/rules/client-event-listeners.md::main", "skill"),
            ("C:/K3_Firehose/.agent/skills/hestiacp-devops/knowledge/05-hestia-database-guide.md::main", "skill"),
            # Brain artifacts (including filenames with 'skill', 'kickoff' -> '/ki', 'research', 'governance')
            ("brain:21d5787a-9886-4798-99d0-046410654cae/implementation_plan_skills_ingestion.md::chunk[0]", "brain"),
            ("brain:6478a577/kickoff.md::main", "brain"),
            ("brain:0bd7983d-02b0-442f-bdd9-f6f767f41350/implementation_plan_governance.md::main", "brain"),
            ("brain:app5_audit.md::main", "brain"),
            # Research (deep-research, deep-result, night3, nlm, tessa-manda, .docx with ::chunk[n]/::main)
            ("deep-research:Adapting_Real_Estate_SEO_to_AI_Overviews.md::chunk[0]", "research"),
            ("deep-result:AI Ghostwriter Prose Analysis Request.md::chunk[0]", "research"),
            ("night3:AI_MLS_Compliance_Auditor_Research.md::chunk[0]", "research"),
            ("nlm:2026 AI Stack Evaluation Brief.md::chunk[0]", "research"),
            ("nlm:Skill Building Guide.md::main", "research"),
            ("tessa-manda:01_deal_structures.md::chunk[0]", "research"),
            ("Digital Nomads Buying NE Florida Homes.docx::chunk[0]", "research"),
            ("Unrelated Topic Analysis.docx::main", "research"),
            ("Unrelated Topic Analysis.docx::chunk[12]", "research"),
            # Rules (KI:k3-root/AGENTS.md, .agents/rules/*, governance:*, proto-agent:rules/*)
            ("KI:k3-root/AGENTS.md::chunk[0]", "rules"),
            ("KI:k3-root/.agents/rules/architecture-boundaries.md::chunk[0]", "rules"),
            ("KI:k3-root/.agents/rules/model-routing.md::main", "rules"),
            (".agents/rules/context-hygiene.md::main", "rules"),
            ("governance:AGENTS.md::chunk[0]", "rules"),
            ("proto-agent:rules/kill-list.prompt.md::main", "rules"),
            # Knowledge (KI:*, plus KI:k3-root MEMORY.md, pitfalls.md, blueprints, docs)
            ("KI:k3-root/.agents/MEMORY.md::main", "knowledge"),
            ("KI:k3-root/.agents/pitfalls.md::chunk[0]", "knowledge"),
            ("KI:k3-root/.agents/blueprints/phase3-governance-lint.md::chunk[0]", "knowledge"),
            ("KI:k3-root/docs/Gemini_4_Argon_Antigravity_Spec.md::chunk[0]", "knowledge"),
            ("KI:32095_brokerage_platform/32095_brokerage_platform_deep_research.md::main", "knowledge"),
            ("KI:agentic_reasoning_and_metacognition/overview.md::main", "knowledge"),
            # Book (proto-agent, proto-script, proto-book, forge-ref, extraction, vault, template)
            ("proto-agent:MEMORY.md::chunk[0]", "book"),
            ("proto-agent:context/reference-corpus/novel_research_notes.md::chunk[0]", "book"),
            ("proto-script:compile-epub.js::chunk[0]", "book"),
            ("proto-book:ai-caching-for-novel-drafting.md::chunk[0]", "book"),
            ("forge-ref:agentic/skill-building.md::chunk[0]", "book"),
            ("extraction:(Ball & Chain #1) Jayne Kingston - Mr. Sir_extraction::main", "book"),
            ("vault:western/ai-western-romance-series-bible-design.md::chunk[0]", "book"),
            ("template:beat-sheet.json::main", "book"),
        ]
        for key, expected in cases:
            with self.subTest(key=key):
                self.assertEqual(mrl._classify_key(key), expected)

    def _build_fixture(self):
        today = datetime.date.today()
        recent_str = (today - datetime.timedelta(days=2)).isoformat()
        old_str = (today - datetime.timedelta(days=90)).isoformat()

        vec_high = np.zeros(768, dtype=np.float32)
        vec_high[0] = 1.0
        vec_mid = np.zeros(768, dtype=np.float32)
        vec_mid[0] = 0.8
        vec_mid[1] = 0.6

        index = {
            "mcp:graphify.query_graph::main": {
                "vector": vec_high,
                "hash": "h1",
                "snippet": "MCP tool graphify.query_graph",
                "mtime": recent_str,
            },
            "skill:code-reviewer::main": {
                "vector": vec_high,
                "hash": "h2",
                "snippet": "Skill code-reviewer",
                "mtime": recent_str,
            },
            "KI:k3-root/.agents/rules/model-routing.md::main": {
                "vector": vec_high,
                "hash": "h3",
                "snippet": "Model routing rule",
                "mtime": recent_str,
            },
            "brain:6478a577/walkthrough.md::main": {
                "vector": vec_mid,
                "hash": "h4",
                "snippet": "Brain walkthrough recent",
                "mtime": recent_str,
            },
            "KI:k3-root/.agents/pitfalls.md::chunk[0]": {
                "vector": vec_mid,
                "hash": "h5",
                "snippet": "Pitfall log old entry",
                "mtime": old_str,
            },
            "deep-research:Agent_Memory.md::chunk[0]": {
                "vector": vec_mid,
                "hash": "h6",
                "snippet": "Deep research memory",
                "mtime": recent_str,
            },
            "proto-book:novel-guide.md::main": {
                "vector": vec_mid,
                "hash": "h7",
                "snippet": "Proto book guide",
            },
        }
        keys = list(index.keys())
        matrix = np.vstack([index[k]["vector"] for k in keys]).astype(np.float32)
        return index, matrix, keys, vec_high, recent_str, old_str

    def test_default_exclusions_and_include_all(self):
        index, matrix, keys, q_vec, _, _ = self._build_fixture()
        with mock.patch.object(mrl, "_ensure_index", return_value=(index, matrix, keys)), \
             mock.patch.object(mrl, "_embed_query", return_value=q_vec):
            out_default = mrl.mrl_search("test query", top_k=10)
            self.assertNotIn("[TOOL]", out_default)
            self.assertNotIn("[SKILL]", out_default)
            self.assertNotIn("[RULES]", out_default)
            self.assertIn("[BRAIN]", out_default)
            self.assertIn("[KNOWLEDGE]", out_default)
            self.assertIn("[RESEARCH]", out_default)
            self.assertIn("[BOOK]", out_default)

            out_all = mrl.mrl_search("test query", top_k=10, include_all=True)
            self.assertIn("[TOOL]", out_all)
            self.assertIn("[SKILL]", out_all)
            self.assertIn("[RULES]", out_all)

            out_filter_all = mrl.mrl_search("test query", top_k=10, filter_type="all")
            self.assertIn("[TOOL]", out_filter_all)
            self.assertIn("[SKILL]", out_filter_all)
            self.assertIn("[RULES]", out_filter_all)

    def test_explicit_filter_types(self):
        index, matrix, keys, q_vec, _, _ = self._build_fixture()
        with mock.patch.object(mrl, "_ensure_index", return_value=(index, matrix, keys)), \
             mock.patch.object(mrl, "_embed_query", return_value=q_vec):
            for ftype in ("tool", "skill", "rules", "brain", "knowledge", "research", "book"):
                out = mrl.mrl_search("test query", top_k=10, filter_type=ftype)
                self.assertIn(f"[{ftype.upper()}]", out)
                for other in ("tool", "skill", "rules", "brain", "knowledge", "research", "book"):
                    if other != ftype:
                        self.assertNotIn(f"[{other.upper()}]", out)

            out_skills = mrl.mrl_search_skills("test query", top_k=5)
            self.assertIn("[SKILL]", out_skills)
            self.assertNotIn("[BRAIN]", out_skills)

    def test_mtime_formatting_and_max_age_days(self):
        index, matrix, keys, q_vec, recent_str, old_str = self._build_fixture()
        with mock.patch.object(mrl, "_ensure_index", return_value=(index, matrix, keys)), \
             mock.patch.object(mrl, "_embed_query", return_value=q_vec):
            out = mrl.mrl_search("test query", top_k=10)
            self.assertIn(f"[mtime: {recent_str}]", out)
            self.assertIn(f"[mtime: {old_str}]", out)

            out_recent = mrl.mrl_search("test query", top_k=10, max_age_days=14)
            self.assertIn(f"[mtime: {recent_str}]", out_recent)
            self.assertNotIn(f"[mtime: {old_str}]", out_recent)
            self.assertNotIn("proto-book:novel-guide.md::main", out_recent)

    def test_contrastive_docstrings(self):
        for fn in (mrl.mrl_search, mrl.mrl_search_skills, mrl.mrl_index_stats):
            doc = fn.__doc__ or ""
            self.assertIn("Use INSTEAD OF", doc)
            self.assertIn("Do NOT use for:", doc)
            self.assertNotIn("25,000+", doc)


if __name__ == "__main__":
    unittest.main()
