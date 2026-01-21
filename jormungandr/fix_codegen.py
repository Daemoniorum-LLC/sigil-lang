import sys

with open("/home/lilith/development/workspace/sigil/sigil-lang/self-hosted/src/codegen.sg", "r") as f:
    content = f.read()

# Replace push_char with push
content = content.replace("push_char(32)", "push(' ')")
content = content.replace("push_char(40)", "push('(')")

with open("/home/lilith/development/workspace/sigil/sigil-lang/self-hosted/src/codegen.sg", "w") as f:
    f.write(content)
print("Fixed push_char to push")
