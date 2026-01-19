#!/usr/bin/env python3
"""
Analyze Sigil stdlib module with Samael/Infernum
Usage: ./analyze_stdlib.py <module_name>
Example: ./analyze_stdlib.py string
"""

import sys
import json
import subprocess
import re

STDLIB = "/home/crook/dev/sigil-lang/parser/src/stdlib.rs"
INFERNUM_URL = "http://127.0.0.1:8081/v1/chat/completions"

def extract_module(module_name: str) -> str:
    """Extract the register_<module> function from stdlib.rs"""
    with open(STDLIB, 'r') as f:
        content = f.read()

    # Find the function start
    pattern = rf'^fn register_{module_name}\(interp: &mut Interpreter\) \{{'
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        return ""

    start = match.start()

    # Find matching brace
    depth = 0
    in_string = False
    i = match.end() - 1  # Start at the opening brace

    for j, c in enumerate(content[i:], i):
        if c == '"' and (j == 0 or content[j-1] != '\\'):
            in_string = not in_string
        if not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return content[start:j+1]

    # Fallback: just take 300 lines
    lines = content[start:].split('\n')[:300]
    return '\n'.join(lines)

def call_infernum(code: str, module: str) -> dict:
    """Send code to Infernum for analysis"""

    system_prompt = """You are Samael, an expert test intelligence agent for the Sigil programming language.

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
}"""

    user_prompt = f"""Analyze this Sigil stdlib module and generate test specifications:

```rust
{code[:4000]}
```

Focus on:
- Function correctness
- Edge cases (empty inputs, overflow, etc.)
- Error handling paths
- Type conversions

Generate comprehensive test specs in JSON format."""

    request = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 4096
    }

    result = subprocess.run(
        ["curl", "-s", "-X", "POST", INFERNUM_URL,
         "-H", "Content-Type: application/json",
         "-d", json.dumps(request)],
        capture_output=True, text=True, timeout=300
    )

    return json.loads(result.stdout)

def main():
    module = sys.argv[1] if len(sys.argv) > 1 else "string"

    print(f"=== Samael Stdlib Analyzer ===")
    print(f"Module: {module}")
    print()

    print(f"Extracting register_{module}...")
    code = extract_module(module)

    if not code:
        print(f"Error: Could not find register_{module} in stdlib.rs")
        sys.exit(1)

    line_count = code.count('\n')
    print(f"Extracted {line_count} lines of code")
    print()

    print("Sending to Infernum (Qwen2.5-7B)...")
    print()

    try:
        response = call_infernum(code, module)
        content = response.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
        tokens = response.get('usage', {}).get('total_tokens', '?')

        print(f"=== Samael Analysis ({tokens} tokens) ===")
        print()
        print(content)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
