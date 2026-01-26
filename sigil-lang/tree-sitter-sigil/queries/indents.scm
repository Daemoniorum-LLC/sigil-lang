; Sigil Tree-sitter Indentation Queries
; =====================================

; Indent after opening braces/brackets
[
  (block)
  (struct_body)
  (enum_body)
  (trait_body)
  (impl_body)
  (match_expression)
  (array_expression)
] @indent

; Dedent at closing braces/brackets
[
  "}"
  "]"
  ")"
] @dedent

; Indent continuation of expressions
(binary_expression) @indent
(pipeline_expression) @indent
