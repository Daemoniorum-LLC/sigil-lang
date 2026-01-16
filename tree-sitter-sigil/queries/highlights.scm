; Sigil Tree-sitter Highlighting Queries
; =======================================

; === Comments ===
(line_comment) @comment
(block_comment) @comment

; === Keywords ===
[
  "fn"
  "let"
  "mut"
  "const"
  "type"
  "struct"
  "sigil"
  "enum"
  "trait"
  "impl"
  "mod"
  "scroll"
  "use"
  "invoke"
  "pub"
  "async"
  "move"
  "ref"
  "unsafe"
  "extern"
  "dyn"
  "where"
  "as"
  "macro"
  "rune"
] @keyword

; Control flow
[
  "if"
  "else"
  "match"
  "loop"
  "while"
  "for"
  "in"
  "break"
  "continue"
  "return"
] @keyword.control

; Self/Super/Crate
[
  "self"
  "super"
  "crate"
] @variable.builtin

; === Types ===
(primitive_type) @type.builtin

(struct_definition
  name: (identifier) @type)
(enum_definition
  name: (identifier) @type)
(trait_definition
  name: (identifier) @type)
(type_alias
  name: (identifier) @type)
(generic_type
  (path) @type)

; === Functions ===
(function_definition
  name: (identifier) @function)
(function_signature
  name: (identifier) @function)

; Call expressions - the first child is the function being called
(call_expression
  (path) @function.call)

(method_call
  method: (identifier) @function.method)

; === Variables ===
(parameter
  name: (identifier) @variable.parameter)
(closure_parameter
  (identifier_pattern
    (identifier) @variable.parameter))

(let_statement
  (identifier_pattern
    (identifier) @variable))

(field_expression
  field: (identifier) @property)
(field_initializer
  name: (identifier) @property)
(struct_field
  name: (identifier) @property)

; === Literals ===
(number_literal) @number
(string_literal) @string
(char_literal) @character
(boolean_literal) @boolean
(escape_sequence) @string.escape

; === Operators ===
[
  "+"
  "-"
  "*"
  "/"
  "%"
  "="
  "=="
  "!="
  "<"
  ">"
  "<="
  ">="
  "&&"
  "||"
  "&"
  "|"
  "^"
  "<<"
  ">>"
  "+="
  "-="
  "*="
  "/="
  "%="
  "&="
  "|="
  "^="
  "<<="
  ">>="
  ".."
  "..="
  "=>"
  "->"
  "::"
] @operator

; === Punctuation ===
["(" ")" "[" "]" "{" "}"] @punctuation.bracket
["," ";" ":" "."] @punctuation.delimiter

; === Morphemes (Sigil-specific) ===
(morpheme) @function.builtin

; Greek letter morphemes with special highlighting
[
  "τ" "Τ"
  "φ" "Φ"
  "σ" "Σ"
  "ρ" "Ρ"
  "λ" "Λ"
  "Π"
  "δ" "Δ"
  "ε"
  "ω" "Ω"
  "α"
  "ζ"
  "μ" "Μ"
  "χ" "Χ"
  "ν" "Ν"
  "ξ" "Ξ"
  "ψ" "Ψ"
  "θ" "Θ"
  "κ" "Κ"
  "⌛"
  "∥"
  "⊛"
] @function.builtin

; ASCII morpheme names
[
  "tau"
  "phi"
  "sigma"
  "rho"
  "lambda"
  "delta"
  "epsilon"
  "omega"
  "alpha"
  "zeta"
  "parallel"
  "gpu"
  "validate"
] @function.builtin

; === Evidentiality Markers (Sigil-specific) ===
(evidentiality_marker) @type.qualifier

; Evidentiality as type modifier - distinct highlighting
(evidentiality_type
  (evidentiality_marker) @type.qualifier)

; Known marker
(evidentiality_marker
  "!" @constant.builtin)

; Uncertain marker
(evidentiality_marker
  "?" @constant.builtin)

; Reported marker (external data)
(evidentiality_marker
  "~" @constant.builtin)

; Paradox marker
(evidentiality_marker
  "‽" @constant.builtin)

; === Pipeline ===
(pipeline_expression
  "|" @operator.special)

; === Lifetimes ===
(lifetime) @label

; === Labels ===
(label) @label

; === Attributes ===
; (attribute) @attribute

; === Special identifiers ===
((identifier) @constant
  (#match? @constant "^[A-Z][A-Z_0-9]+$"))

; Underscore placeholder in closures/morphemes
((identifier) @variable.builtin
  (#eq? @variable.builtin "_"))

; === Errors ===
(ERROR) @error
