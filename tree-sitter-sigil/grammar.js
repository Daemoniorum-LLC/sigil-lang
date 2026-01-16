/**
 * Tree-sitter grammar for the Sigil programming language
 *
 * Sigil is a polysynthetic programming language with evidentiality types,
 * morpheme operators, and AI-native features.
 */

module.exports = grammar({
  name: 'sigil',

  extras: $ => [
    /\s/,
    $.line_comment,
    $.block_comment,
  ],

  word: $ => $.identifier,

  conflicts: $ => [
    // The | operator is used for both bitwise OR and pipeline expressions
    [$.binary_expression, $.pipeline_expression],
    [$.binary_expression, $.pipeline_expression, $.range_expression],
    // field_expression and method_call both start with expr.ident
    [$.field_expression, $.method_call],
    // Loop expressions can have optional labels which conflict with break label
    [$.loop_expression],
    [$.while_expression],
    [$.for_expression],
  ],

  // Allow struct expressions in expression position (like Rust's block expressions)
  inline: $ => [
  ],

  rules: {
    // === Entry Point ===
    source_file: $ => repeat($._item),

    _item: $ => choice(
      $.function_definition,
      $.struct_definition,
      $.enum_definition,
      $.trait_definition,
      $.impl_block,
      $.use_declaration,
      $.mod_declaration,
      $.const_declaration,
      $.type_alias,
      $.macro_definition,
    ),

    // === Comments ===
    line_comment: $ => token(seq('//', /.*/)),

    // Simple block comment (non-nested)
    block_comment: $ => token(seq('/*', /[^*]*\*+([^/*][^*]*\*+)*/, '/')),

    doc_comment: $ => token(seq('//!', /.*/)),

    // === Function Definition ===
    function_definition: $ => seq(
      optional($.visibility),
      optional('async'),
      'fn',
      field('name', $.identifier),
      optional($.generic_parameters),
      $.parameter_list,
      optional(seq('->', $._type)),
      optional($.where_clause),
      $.block,
    ),

    parameter_list: $ => seq(
      '(',
      optional(seq(
        $.parameter,
        repeat(seq(',', $.parameter)),
        optional(','),
      )),
      ')',
    ),

    parameter: $ => seq(
      optional('mut'),
      field('name', $.identifier),
      optional($.evidentiality_marker),
      optional(seq(':', $._type)),
    ),

    // === Struct Definition ===
    struct_definition: $ => seq(
      optional($.visibility),
      choice('struct', 'sigil'),
      field('name', $.identifier),
      optional($.generic_parameters),
      optional($.where_clause),
      choice(
        $.struct_body,
        seq($.tuple_fields, ';'),
        ';',
      ),
    ),

    struct_body: $ => seq(
      '{',
      optional(seq(
        $.struct_field,
        repeat(seq(',', $.struct_field)),
        optional(','),
      )),
      '}',
    ),

    struct_field: $ => seq(
      optional($.visibility),
      field('name', $.identifier),
      ':',
      $._type,
    ),

    tuple_fields: $ => seq(
      '(',
      optional(seq(
        $._type,
        repeat(seq(',', $._type)),
        optional(','),
      )),
      ')',
    ),

    // === Enum Definition ===
    enum_definition: $ => seq(
      optional($.visibility),
      'enum',
      field('name', $.identifier),
      optional($.generic_parameters),
      optional($.where_clause),
      $.enum_body,
    ),

    enum_body: $ => seq(
      '{',
      optional(seq(
        $.enum_variant,
        repeat(seq(',', $.enum_variant)),
        optional(','),
      )),
      '}',
    ),

    enum_variant: $ => seq(
      field('name', $.identifier),
      optional(choice(
        $.tuple_fields,
        $.struct_body,
        seq('=', $._expression),
      )),
    ),

    // === Trait Definition ===
    trait_definition: $ => seq(
      optional($.visibility),
      'trait',
      field('name', $.identifier),
      optional($.generic_parameters),
      optional($.trait_bounds),
      optional($.where_clause),
      $.trait_body,
    ),

    trait_body: $ => seq(
      '{',
      repeat(choice(
        $.function_definition,
        $.function_signature,
        $.type_alias,
        $.const_declaration,
      )),
      '}',
    ),

    function_signature: $ => seq(
      optional($.visibility),
      optional('async'),
      'fn',
      field('name', $.identifier),
      optional($.generic_parameters),
      $.parameter_list,
      optional(seq('->', $._type)),
      optional($.where_clause),
      ';',
    ),

    // === Impl Block ===
    impl_block: $ => seq(
      'impl',
      optional($.generic_parameters),
      optional(seq($._type, 'for')),
      $._type,
      optional($.where_clause),
      $.impl_body,
    ),

    impl_body: $ => seq(
      '{',
      repeat(choice(
        $.function_definition,
        $.type_alias,
        $.const_declaration,
      )),
      '}',
    ),

    // === Use Declaration ===
    use_declaration: $ => seq(
      optional($.visibility),
      choice('use', 'invoke'),
      $.use_path,
      ';',
    ),

    use_path: $ => seq(
      optional(seq('::', optional('crate'))),
      $.identifier,
      repeat(seq('::', choice(
        $.identifier,
        '*',
        $.use_group,
      ))),
      optional(seq('as', $.identifier)),
    ),

    use_group: $ => seq(
      '{',
      optional(seq(
        $.use_path,
        repeat(seq(',', $.use_path)),
        optional(','),
      )),
      '}',
    ),

    // === Module Declaration ===
    mod_declaration: $ => seq(
      optional($.visibility),
      choice('mod', 'scroll'),
      field('name', $.identifier),
      choice(
        ';',
        $.block,
      ),
    ),

    // === Const Declaration ===
    const_declaration: $ => seq(
      optional($.visibility),
      'const',
      field('name', $.identifier),
      ':',
      $._type,
      '=',
      $._expression,
      ';',
    ),

    // === Type Alias ===
    type_alias: $ => seq(
      optional($.visibility),
      'type',
      field('name', $.identifier),
      optional($.generic_parameters),
      '=',
      $._type,
      ';',
    ),

    // === Macro Definition ===
    macro_definition: $ => seq(
      optional($.visibility),
      choice('macro', 'rune'),
      field('name', $.identifier),
      $.macro_body,
    ),

    macro_body: $ => seq(
      '{',
      // Simplified: just capture tokens until closing brace
      repeat(choice(
        $.identifier,
        $.string_literal,
        $.number_literal,
        /[^{}]+/,
        seq('{', repeat(/[^{}]*/), '}'),
      )),
      '}',
    ),

    // === Visibility ===
    visibility: $ => choice(
      'pub',
      seq('pub', '(', choice('crate', 'super', seq('in', $.path)), ')'),
    ),

    // === Generics ===
    generic_parameters: $ => seq(
      '<',
      optional(seq(
        $.generic_parameter,
        repeat(seq(',', $.generic_parameter)),
        optional(','),
      )),
      '>',
    ),

    generic_parameter: $ => choice(
      $.type_parameter,
      $.lifetime_parameter,
      $.const_parameter,
    ),

    type_parameter: $ => seq(
      field('name', $.identifier),
      optional($.trait_bounds),
      optional(seq('=', $._type)),
    ),

    lifetime_parameter: $ => seq(
      $.lifetime,
      optional(seq(':', $.lifetime_bounds)),
    ),

    const_parameter: $ => seq(
      'const',
      field('name', $.identifier),
      ':',
      $._type,
    ),

    trait_bounds: $ => prec.right(seq(
      ':',
      $.trait_bound,
      repeat(seq('+', $.trait_bound)),
    )),

    trait_bound: $ => choice(
      $._type,
      $.lifetime,
    ),

    lifetime_bounds: $ => seq(
      $.lifetime,
      repeat(seq('+', $.lifetime)),
    ),

    where_clause: $ => seq(
      'where',
      $.where_predicate,
      repeat(seq(',', $.where_predicate)),
      // No trailing comma - causes ambiguity with tuple structs
    ),

    where_predicate: $ => seq(
      choice($._type, $.lifetime),
      ':',
      choice($.trait_bounds, $.lifetime_bounds),
    ),

    // === Types ===
    // Compound types like references, pointers, arrays can contain any type
    // Evidentiality types only apply to simple/named types to avoid ambiguity
    _type: $ => choice(
      $._simple_type,
      $.reference_type,
      $.pointer_type,
      $.array_type,
      $.slice_type,
      $.tuple_type,
      $.function_type,
      $.impl_trait,
      $.dyn_trait,
    ),

    // Simple types that can have evidentiality markers applied
    _simple_type: $ => choice(
      $.primitive_type,
      $.path,
      $.generic_type,
      $.inferred_type,
      $.never_type,
      $.evidentiality_type,
    ),

    primitive_type: $ => choice(
      'i8', 'i16', 'i32', 'i64', 'i128', 'isize',
      'u8', 'u16', 'u32', 'u64', 'u128', 'usize',
      'f32', 'f64',
      'bool', 'char', 'str',
    ),

    generic_type: $ => prec(1, seq(
      $.path,
      $.type_arguments,
    )),

    type_arguments: $ => seq(
      '<',
      optional(seq(
        $._type,
        repeat(seq(',', $._type)),
        optional(','),
      )),
      '>',
    ),

    reference_type: $ => seq(
      '&',
      optional($.lifetime),
      optional('mut'),
      $._type,
    ),

    pointer_type: $ => seq(
      '*',
      choice('const', 'mut'),
      $._type,
    ),

    array_type: $ => seq(
      '[',
      $._type,
      ';',
      $._expression,
      ']',
    ),

    slice_type: $ => seq(
      '[',
      $._type,
      ']',
    ),

    tuple_type: $ => seq(
      '(',
      optional(seq(
        $._type,
        repeat(seq(',', $._type)),
        optional(','),
      )),
      ')',
    ),

    function_type: $ => seq(
      optional('unsafe'),
      optional(seq('extern', optional($.string_literal))),
      'fn',
      $.parameter_list,
      optional(seq('->', $._type)),
    ),

    impl_trait: $ => seq(
      'impl',
      $.trait_bounds,
    ),

    dyn_trait: $ => seq(
      'dyn',
      $.trait_bounds,
    ),

    inferred_type: $ => '_',

    never_type: $ => '!',

    // === Evidentiality Types (Sigil-specific) ===
    // Evidentiality markers apply to simple named types, not compound types
    // e.g., Result!, Option?, but not &i32! (use &(i32!) if needed)
    // prec(1) ensures we greedily consume the evidentiality marker
    evidentiality_type: $ => prec(1, seq(
      choice(
        $.primitive_type,
        $.path,
        $.generic_type,
      ),
      $.evidentiality_marker,
    )),

    evidentiality_marker: $ => choice(
      '!',   // Known
      '?',   // Uncertain
      '~',   // Reported
      '‽',   // Paradox (interrobang)
    ),

    // === Statements ===
    block: $ => seq(
      '{',
      repeat($._statement),
      optional($._expression),
      '}',
    ),

    _statement: $ => choice(
      $.let_statement,
      $.expression_statement,
      $.return_statement,
      $.break_statement,
      $.continue_statement,
      $.assignment_statement,
      $._item,
    ),

    let_statement: $ => seq(
      'let',
      // Note: 'mut' is part of identifier_pattern, not here
      $._pattern,
      optional($.evidentiality_marker),
      optional(seq(':', $._type)),
      optional(seq('=', $._expression)),
      ';',
    ),

    expression_statement: $ => seq(
      $._expression,
      ';',
    ),

    return_statement: $ => seq(
      'return',
      optional($._expression),
      ';',
    ),

    break_statement: $ => seq(
      'break',
      optional($.label),
      optional($._expression),
      ';',
    ),

    continue_statement: $ => seq(
      'continue',
      optional($.label),
      ';',
    ),

    assignment_statement: $ => seq(
      $._expression,
      choice('=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='),
      $._expression,
      ';',
    ),

    // === Expressions ===
    // Note: $.identifier is not listed separately since $.path covers single identifiers
    _expression: $ => choice(
      $.path,
      $.literal,
      $.unary_expression,
      $.binary_expression,
      $.call_expression,
      $.method_call,
      $.field_expression,
      $.index_expression,
      $.array_expression,
      $.tuple_expression,
      $.struct_expression,
      $.if_expression,
      $.match_expression,
      $.loop_expression,
      $.while_expression,
      $.for_expression,
      $.block,
      $.closure,
      $.reference_expression,
      $.dereference_expression,
      $.try_expression,
      $.await_expression,
      $.cast_expression,
      $.range_expression,
      $.parenthesized_expression,
      $.pipeline_expression,
      $.morpheme_expression,
    ),

    literal: $ => choice(
      $.number_literal,
      $.string_literal,
      $.char_literal,
      $.boolean_literal,
    ),

    number_literal: $ => token(choice(
      // Decimal
      /[0-9][0-9_]*/,
      // Decimal with suffix
      /[0-9][0-9_]*\.[0-9][0-9_]*/,
      // Hex
      /0x[0-9a-fA-F_]+/,
      // Octal
      /0o[0-7_]+/,
      // Binary
      /0b[01_]+/,
      // Float with exponent
      /[0-9][0-9_]*[eE][+-]?[0-9_]+/,
      /[0-9][0-9_]*\.[0-9][0-9_]*[eE][+-]?[0-9_]+/,
    )),

    string_literal: $ => choice(
      seq('"', repeat(choice(/[^"\\]/, $.escape_sequence)), '"'),
      seq('r#"', /[^"]*/, '"#'),
      seq('r##"', /[^"]*/, '"##'),
    ),

    char_literal: $ => seq("'", choice(/[^'\\]/, $.escape_sequence), "'"),

    escape_sequence: $ => token(seq('\\', choice(
      /['"\\nrt0]/,
      /x[0-9a-fA-F]{2}/,
      /u\{[0-9a-fA-F]+\}/,
    ))),

    boolean_literal: $ => choice('true', 'false'),

    unary_expression: $ => prec.left(14, choice(
      seq('-', $._expression),
      seq('!', $._expression),
      // Note: * and & are handled by dereference_expression and reference_expression
    )),

    binary_expression: $ => choice(
      // Arithmetic
      prec.left(11, seq($._expression, '*', $._expression)),
      prec.left(11, seq($._expression, '/', $._expression)),
      prec.left(11, seq($._expression, '%', $._expression)),
      prec.left(10, seq($._expression, '+', $._expression)),
      prec.left(10, seq($._expression, '-', $._expression)),
      // Shift
      prec.left(9, seq($._expression, '<<', $._expression)),
      prec.left(9, seq($._expression, '>>', $._expression)),
      // Bitwise
      prec.left(8, seq($._expression, '&', $._expression)),
      prec.left(7, seq($._expression, '^', $._expression)),
      prec.left(6, seq($._expression, '|', $._expression)),
      // Comparison
      prec.left(5, seq($._expression, '==', $._expression)),
      prec.left(5, seq($._expression, '!=', $._expression)),
      prec.left(5, seq($._expression, '<', $._expression)),
      prec.left(5, seq($._expression, '>', $._expression)),
      prec.left(5, seq($._expression, '<=', $._expression)),
      prec.left(5, seq($._expression, '>=', $._expression)),
      // Logical
      prec.left(4, seq($._expression, '&&', $._expression)),
      prec.left(3, seq($._expression, '||', $._expression)),
    ),

    call_expression: $ => prec(15, seq(
      $._expression,
      '(',
      optional(seq(
        $._expression,
        repeat(seq(',', $._expression)),
        optional(','),
      )),
      ')',
    )),

    method_call: $ => prec(15, seq(
      $._expression,
      '.',
      field('method', $.identifier),
      optional($.type_arguments),
      '(',
      optional(seq(
        $._expression,
        repeat(seq(',', $._expression)),
        optional(','),
      )),
      ')',
    )),

    field_expression: $ => prec(15, seq(
      $._expression,
      '.',
      field('field', choice($.identifier, $.number_literal)),
    )),

    index_expression: $ => prec(15, seq(
      $._expression,
      '[',
      $._expression,
      ']',
    )),

    array_expression: $ => seq(
      '[',
      choice(
        // Array literal
        optional(seq(
          $._expression,
          repeat(seq(',', $._expression)),
          optional(','),
        )),
        // Array repeat
        seq($._expression, ';', $._expression),
      ),
      ']',
    ),

    tuple_expression: $ => seq(
      '(',
      optional(seq(
        $._expression,
        ',',
        repeat(seq($._expression, ',')),
        optional($._expression),
      )),
      ')',
    ),

    // Lower precedence than path so `if cond { }` isn't parsed as struct literal
    struct_expression: $ => prec(-1, seq(
      $.path,
      '{',
      optional(seq(
        $.field_initializer,
        repeat(seq(',', $.field_initializer)),
        optional(','),
      )),
      optional(seq('..', $._expression)),
      '}',
    )),

    field_initializer: $ => choice(
      seq(field('name', $.identifier), ':', $._expression),
      $.identifier,
    ),

    if_expression: $ => prec.right(seq(
      'if',
      $._expression,
      $.block,
      optional(seq('else', choice($.block, $.if_expression))),
    )),

    match_expression: $ => seq(
      'match',
      $._expression,
      '{',
      optional(seq(
        $.match_arm,
        repeat(seq(',', $.match_arm)),
        optional(','),
      )),
      '}',
    ),

    match_arm: $ => seq(
      $._pattern,
      optional(seq('if', $._expression)),
      '=>',
      $._expression,  // _expression includes block
    ),

    loop_expression: $ => seq(
      optional($.label),
      'loop',
      $.block,
    ),

    while_expression: $ => seq(
      optional($.label),
      'while',
      $._expression,
      $.block,
    ),

    for_expression: $ => seq(
      optional($.label),
      'for',
      $._pattern,
      'in',
      $._expression,
      $.block,
    ),

    label: $ => seq("'", $.identifier, ':'),

    closure: $ => seq(
      optional('move'),
      choice(
        seq('|', optional($.closure_parameters), '|'),
        '||',
      ),
      optional(seq('->', $._type)),
      $._expression,  // _expression includes block
    ),

    closure_parameters: $ => seq(
      $.closure_parameter,
      repeat(seq(',', $.closure_parameter)),
      optional(','),
    ),

    closure_parameter: $ => prec(2, seq(
      // Note: 'mut' is part of identifier_pattern, not here
      $._pattern,
      optional(seq(':', $._type)),
    )),

    reference_expression: $ => prec(14, seq(
      '&',
      optional('mut'),
      $._expression,
    )),

    dereference_expression: $ => prec(14, seq(
      '*',
      $._expression,
    )),

    try_expression: $ => prec(15, seq(
      $._expression,
      '?',
    )),

    await_expression: $ => prec(15, choice(
      seq($._expression, '.', 'await'),
      seq($._expression, '⌛'),
    )),

    cast_expression: $ => prec.left(12, seq(
      $._expression,
      'as',
      $._type,
    )),

    range_expression: $ => prec.left(2, choice(
      seq($._expression, '..', $._expression),
      seq($._expression, '..=', $._expression),
      seq($._expression, '..'),
      seq('..', $._expression),
      seq('..=', $._expression),
      '..',
    )),

    parenthesized_expression: $ => seq(
      '(',
      $._expression,
      ')',
    ),

    // === Sigil-specific: Pipeline and Morpheme Expressions ===
    pipeline_expression: $ => prec.left(1, seq(
      $._expression,
      '|',
      $.morpheme_expression,
    )),

    // prec.right ensures morpheme greedily consumes the body
    morpheme_expression: $ => prec.right(choice(
      // Morpheme with closure body
      seq($.morpheme, $.morpheme_body),
      // Standalone morpheme (no body)
      $.morpheme,
    )),

    morpheme: $ => choice(
      // Greek letters
      'τ', 'Τ',  // tau - transform/map
      'φ', 'Φ',  // phi - filter
      'σ', 'Σ',  // sigma - sort/sum
      'ρ', 'Ρ',  // rho - reduce
      'λ', 'Λ',  // lambda
      'Π',       // pi - product
      'δ', 'Δ',  // delta - difference
      'ε',       // epsilon - empty
      'ω', 'Ω',  // omega - end
      'α',       // alpha - first
      'ζ',       // zeta - zip
      'μ', 'Μ',  // mu - middle
      'χ', 'Χ',  // chi - random
      'ν', 'Ν',  // nu - nth
      'ξ', 'Ξ',  // xi - next
      'ψ', 'Ψ',  // psi - mental state
      'θ', 'Θ',  // theta - threshold
      'κ', 'Κ',  // kappa - callback
      '⌛',      // hourglass - await
      // ASCII equivalents
      'tau', 'phi', 'sigma', 'rho', 'lambda',
      'delta', 'epsilon', 'omega', 'alpha', 'zeta',
      // Parallel/GPU
      '∥', 'parallel',
      '⊛', 'gpu',
      // Validation morpheme - prec.right ensures it greedily takes the marker
      prec.right(seq('validate', optional($.evidentiality_marker))),
    ),

    morpheme_body: $ => seq(
      '{',
      choice(
        // Simple expression with _ placeholder
        $._expression,
        // Reduce form: initial, acc, x => expr
        seq(
          $._expression, ',',
          $.identifier, ',',
          $.identifier, '=>',
          $._expression,
        ),
      ),
      '}',
    ),

    // === Patterns ===
    _pattern: $ => choice(
      $.identifier_pattern,
      $.wildcard_pattern,
      $.tuple_pattern,
      $.struct_pattern,
      $.enum_pattern,
      $.literal,
      $.range_pattern,
      $.reference_pattern,
      $.or_pattern,
    ),

    identifier_pattern: $ => prec(1, seq(
      optional('ref'),
      optional('mut'),
      $.identifier,
      optional(seq('@', $._pattern)),
    )),

    wildcard_pattern: $ => '_',

    tuple_pattern: $ => seq(
      '(',
      optional(seq(
        $._pattern,
        repeat(seq(',', $._pattern)),
        optional(','),
      )),
      ')',
    ),

    struct_pattern: $ => seq(
      $.path,
      '{',
      optional(seq(
        $.field_pattern,
        repeat(seq(',', $.field_pattern)),
        optional(','),
      )),
      optional('..'),
      '}',
    ),

    field_pattern: $ => choice(
      seq(field('name', $.identifier), ':', $._pattern),
      seq(optional('ref'), optional('mut'), $.identifier),
    ),

    enum_pattern: $ => seq(
      $.path,
      optional(choice(
        $.tuple_pattern,
        $.struct_pattern,
      )),
    ),

    range_pattern: $ => prec(1, seq(
      $.literal,
      choice('..', '..='),
      $.literal,
    )),

    reference_pattern: $ => prec(1, seq(
      '&',
      // Note: 'mut' is part of identifier_pattern, not here
      $._pattern,
    )),

    or_pattern: $ => prec.left(seq(
      $._pattern,
      '|',
      $._pattern,
    )),

    // === Paths and Identifiers ===
    path: $ => prec.left(seq(
      optional(choice('::', 'crate', 'self', 'super')),
      $.identifier,
      repeat(seq('::', $.identifier)),
    )),

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

    lifetime: $ => seq("'", $.identifier),
  },
});
