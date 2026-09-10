#!/usr/bin/env python3
"""Regression tests for the keyword-identifier pass (#82).

The pass used to rename every keyword it found in a binding position, on the
premise that "a keyword can never legitimately be an identifier". It is false:
most keywords are accepted as struct field names, and renaming them rewrote nine
entries of an append-only file to buy nothing.

These tests are written against the REAL compiler — `sigil check` decides what is
accepted, not a table in this file — so they cannot drift away from the language.
They skip, loudly, when no `sigil` binary can be found.

Run:  python3 tools/test_migrate_legacy_syntax.py
      (also collectable by pytest)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import migrate_legacy_syntax as m  # noqa: E402


HAVE_SIGIL = bool(m._probe_toolchain())
NEEDS_SIGIL = unittest.skipUnless(
    HAVE_SIGIL, "no usable `sigil` binary; build parser/ or set SIGIL_BIN"
)

# The five names #82 was filed about. The parser accepts every one of them as a
# struct field, in the declaration and in the literal, so none may be renamed.
# `state` is in the list because the issue names it: it turns out not to be a
# keyword token at all, which the last test below pins down.
ACCEPTED_AS_FIELDS = ["anima", "state", "body", "layer", "scope"]

# The two the pass genuinely exists for: refused in every position tested.
REJECTED_EVERYWHERE = ["aspect", "alter"]


class KeywordSet(unittest.TestCase):
    def test_keywords_come_from_the_lexer_not_a_list(self):
        kws = m.sigil_keywords()
        self.assertIn("aspect", kws)
        self.assertIn("anima", kws)
        # Read out of the token table, so it covers keywords the old hand-written
        # list had simply never been told about.
        for missed_by_the_old_list in ("mut", "loop", "where", "yield", "tome"):
            self.assertIn(missed_by_the_old_list, kws)

    def test_state_is_not_a_keyword(self):
        """#82 calls `state` a keyword token. It is not — `states` is."""
        kws = m.sigil_keywords()
        self.assertNotIn("state", kws)
        self.assertIn("states", kws)


@NEEDS_SIGIL
class AcceptedKeywordsSurvive(unittest.TestCase):
    """The bug: names the parser is perfectly happy with were renamed anyway."""

    def test_field_declaration_and_literal_are_untouched(self):
        src = (
            "Σ Accepted {\n"
            + "".join("    %s: Int,\n" % n for n in ACCEPTED_AS_FIELDS)
            + "}\n\nrite main() {\n    ≔ a = Accepted { "
            + ", ".join("%s: 1" % n for n in ACCEPTED_AS_FIELDS)
            + " };\n}\n"
        )
        self.assertEqual(m.migrate_constructs(src), src)

    def test_each_accepted_name_individually(self):
        for name in ACCEPTED_AS_FIELDS:
            with self.subTest(name=name):
                src = "Σ P {\n    %s: Int,\n}\n" % name
                self.assertEqual(m.migrate_constructs(src), src)
                self.assertNotIn(name + "_", m.migrate_constructs(src))

    def test_conclave_shaped_entry_is_not_rewritten(self):
        """The exact shape the pass damaged: an append-only record's `anima:`."""
        src = (
            "acolyte : Reflecting {\n"
            "    anima: AnimaState {\n"
            "        pleasure: 0.7~, arousal: 0.4~, dominance: 0.6~,\n"
            "    },\n"
            "}\n"
        )
        self.assertEqual(m.migrate_constructs(src), src)

    def test_accepted_names_are_also_left_alone_as_parameters(self):
        for name in ("anima", "body", "layer", "scope"):
            with self.subTest(name=name):
                src = "rite f(%s: Int) -> Int {\n    ret 1;\n}\n" % name
                self.assertEqual(m.migrate_constructs(src), src)


@NEEDS_SIGIL
class RejectedKeywordsAreStillRenamed(unittest.TestCase):
    """The pass must keep doing the job it was written for."""

    def test_field_declaration(self):
        for name in REJECTED_EVERYWHERE:
            with self.subTest(name=name):
                out = m.migrate_constructs("Σ P {\n    %s: Int,\n}\n" % name)
                self.assertIn(name + "_: Int", out)

    def test_parameter(self):
        for name in REJECTED_EVERYWHERE:
            with self.subTest(name=name):
                out = m.migrate_constructs(
                    "rite f(%s: Int) -> Int {\n    ret %s;\n}\n" % (name, name)
                )
                self.assertIn("f(%s_: Int)" % name, out)
                self.assertIn("ret %s_;" % name, out)

    def test_local_binding(self):
        for name in REJECTED_EVERYWHERE:
            with self.subTest(name=name):
                out = m.migrate_constructs("rite f() {\n    ≔ %s = 1;\n}\n" % name)
                self.assertIn("≔ %s_ = 1;" % name, out)

    def test_rename_makes_the_file_parse(self):
        """A rename is only justified when it buys parseability. Prove it does."""
        sigil = m._probe_toolchain()
        for name in REJECTED_EVERYWHERE:
            with self.subTest(name=name):
                src = "Σ P {\n    %s: Int,\n}\n" % name
                self.assertFalse(m._parses(sigil, src))
                self.assertTrue(m._parses(sigil, m.migrate_constructs(src)))


@NEEDS_SIGIL
class PositionMatters(unittest.TestCase):
    """Acceptance differs by position, and in both directions."""

    def test_a_keyword_can_be_refused_as_a_field_yet_fine_as_a_local(self):
        # `self`, `true` and friends name nothing, but bind fine.
        self.assertTrue(m._rejects("true", "field"))
        self.assertFalse(m._rejects("true", "param"))
        self.assertFalse(m._rejects("true", "local"))

    def test_a_keyword_can_be_fine_as_a_field_yet_refused_as_a_parameter(self):
        # The opposite direction — this is why one global set cannot work.
        for name in ("ref", "super", "tome"):
            with self.subTest(name=name):
                self.assertFalse(m._rejects(name, "field"))
                self.assertTrue(m._rejects(name, "param"))
                self.assertTrue(m._rejects(name, "binder"))

    def test_a_keyword_can_bind_and_still_not_be_readable(self):
        # `≔ const = 1;` parses; `const + 1` does not. The local probe binds and
        # reads for exactly this reason.
        sigil = m._probe_toolchain()
        self.assertTrue(m._parses(sigil, "rite f() {\n    ≔ const = 1;\n}\n"))
        self.assertTrue(m._rejects("const", "local"))

    def test_ref_as_a_field_is_kept_but_as_a_parameter_is_renamed(self):
        field = "Σ P {\n    ref: Int,\n}\n"
        self.assertEqual(m.migrate_constructs(field), field)
        out = m.migrate_constructs("rite f(ref: Int) -> Int {\n    ret 1;\n}\n")
        self.assertIn("f(ref_: Int)", out)


@NEEDS_SIGIL
class NoEvidenceNoRename(unittest.TestCase):
    def test_pass_is_skipped_when_the_compiler_cannot_be_consulted(self):
        """Without the parser there is no evidence, so nothing may be renamed."""
        saved_toolchain, saved_cache = m._TOOLCHAIN, dict(m._REJECT_CACHE)
        m._TOOLCHAIN, m._REJECT_CACHE = False, {}
        try:
            src = "Σ P {\n    aspect: Int,\n}\n"
            self.assertEqual(m.migrate_constructs(src), src)
        finally:
            m._TOOLCHAIN, m._REJECT_CACHE = saved_toolchain, saved_cache


if __name__ == "__main__":
    if not HAVE_SIGIL:
        print(
            "warning: no `sigil` binary found — the parser-backed tests will skip.\n"
            "         build with `cargo build --release` in parser/, or set SIGIL_BIN.",
            file=sys.stderr,
        )
    unittest.main(verbosity=2)
