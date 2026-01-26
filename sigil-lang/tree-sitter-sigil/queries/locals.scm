; Sigil Tree-sitter Locals Queries
; =================================
; Defines scopes and variable definitions for semantic analysis

; Scopes
(function_definition) @local.scope
(closure) @local.scope
(block) @local.scope
(for_expression) @local.scope
(while_expression) @local.scope
(loop_expression) @local.scope
(if_expression) @local.scope
(match_arm) @local.scope

; Definitions
(parameter
  name: (identifier) @local.definition)

(let_statement
  (identifier_pattern
    (identifier) @local.definition))

(for_expression
  (identifier_pattern
    (identifier) @local.definition))

(closure_parameter
  (identifier_pattern
    (identifier) @local.definition))

; References
(identifier) @local.reference
