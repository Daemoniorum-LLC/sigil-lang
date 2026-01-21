#!/usr/bin/env bash
# Analyze Sigil stdlib module with Samael/Infernum
# Usage: ./analyze_stdlib.sh <module_name>
# Example: ./analyze_stdlib.sh string

set -e

MODULE="${1:-string}"
STDLIB="/home/crook/dev/sigil-lang/parser/src/stdlib.rs"
INFERNUM_URL="http://localhost:8081/v1/chat/completions"

echo "=== Samael Stdlib Analyzer ==="
echo "Module: $MODULE"
echo ""

# Extract the module's register function
echo "Extracting register_${MODULE}..."

# Use awk to extract the function body
CODE=$(awk "/^fn register_${MODULE}\(/{p=1} p; /^fn register_[a-z]+\(/ && !/^fn register_${MODULE}\(/{if(p) exit}" "$STDLIB" | head -300)

if [ -z "$CODE" ]; then
    echo "Error: Could not find register_${MODULE} in stdlib.rs"
    exit 1
fi

LINE_COUNT=$(echo "$CODE" | wc -l)
echo "Extracted $LINE_COUNT lines of code"
echo ""

# Build the prompt
SYSTEM_PROMPT='You are Samael, an expert test intelligence agent for the Sigil programming language.

Analyze this stdlib module and generate test specifications. For each stdlib function:
1. Identify what it does and its edge cases
2. Generate BDD-style test specs (Given/When/Then)
3. Prioritize by importance

Output a JSON array of test specs like:
{
  "module": "module_name",
  "specs": [
    {
      "function": "function_name",
      "name": "test_description",
      "priority": "high|medium|low",
      "scenario": {
        "given": "precondition",
        "when": "action",
        "then": "expected result"
      }
    }
  ],
  "coverage_summary": "brief summary"
}'

USER_PROMPT="Analyze this Sigil stdlib module and generate test specifications:

\`\`\`rust
$CODE
\`\`\`

Focus on:
- Function correctness
- Edge cases (empty inputs, overflow, etc.)
- Error handling paths
- Type conversions

Generate comprehensive test specs in JSON format."

# Escape for JSON
SYSTEM_JSON=$(echo "$SYSTEM_PROMPT" | jq -Rs .)
USER_JSON=$(echo "$USER_PROMPT" | jq -Rs .)

# Build request
REQUEST=$(cat <<EOF
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "system", "content": $SYSTEM_JSON},
    {"role": "user", "content": $USER_JSON}
  ],
  "temperature": 0.3,
  "max_tokens": 4096
}
EOF
)

echo "Sending to Infernum (Qwen2.5-7B)..."
echo ""

# Call Infernum
RESPONSE=$(curl -s -X POST "$INFERNUM_URL" \
  -H "Content-Type: application/json" \
  -d "$REQUEST")

# Extract the assistant's message
CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content // "Error: No response"')
TOKENS=$(echo "$RESPONSE" | jq -r '.usage.total_tokens // "?"')

echo "=== Samael Analysis (${TOKENS} tokens) ==="
echo ""
echo "$CONTENT"
