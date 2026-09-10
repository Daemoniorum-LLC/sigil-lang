# The committed corpus

Four files, one per shape the scanner has been wrong about. They exist so the
reserved-name property is checked on every PR in milliseconds, rather than
re-argued from a wide sweep nobody reruns.

`collision_corpus_findings_are_exactly_these` in `parser/src/main.rs` asserts
the **exact** findings for each file. A change to the scanner that alters any
of them has to change the expectation too, in the same commit, where a reviewer
sees it.

The wide sweep (`../sweep.sh`) is the evidence for a corpus-scale claim. This is
the regression test.
