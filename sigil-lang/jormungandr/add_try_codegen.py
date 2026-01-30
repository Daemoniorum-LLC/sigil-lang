with open("/home/lilith/development/workspace/sigil/sigil-lang/self-hosted/src/codegen.sg", "r") as f:
    content = f.read()

# Add Try codegen after ExternBlock
old = '''                "sigil_unit()"
            },

            IrOperation::Closure { .. } => {'''

new = '''                "sigil_unit()"
            },

            IrOperation::Try { .. } => {
                // The ? operator: evaluates expr, returns inner if Ok, early-returns Err
                let expr = op.expr;
                let temp = self.fresh_temp();
                let result_temp = self.fresh_temp();

                // Evaluate the inner expression
                let expr_code = self.emit_operation(*expr);
                self.line(format!("SigilValue {} = {};", temp, expr_code));

                // Check if it's Ok or Err
                self.line(format!("SigilValue {};", result_temp));
                self.line(format!("if ({}.tag == TAG_RESULT_ERR) {{", temp));
                self.indent_push();
                // Early return with the error
                self.line(format!("return {};", temp));
                self.indent_pop();
                self.line("} else {");
                self.indent_push();
                // Extract the Ok value
                self.line(format!("{} = (*(SigilValue*){}.v.ptr);", result_temp, temp));
                self.indent_pop();
                self.line("}");

                result_temp
            },

            IrOperation::Closure { .. } => {'''

if old in content:
    content = content.replace(old, new)
    with open("/home/lilith/development/workspace/sigil/sigil-lang/self-hosted/src/codegen.sg", "w") as f:
        f.write(content)
    print("Added Try codegen")
else:
    print("Pattern not found")
