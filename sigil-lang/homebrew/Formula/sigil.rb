class Sigil < Formula
  desc "Polysynthetic programming language with epistemic types for AI"
  homepage "https://github.com/Daemoniorum-LLC/sigil-lang"
  license "LicenseRef-Daemoniorum"
  head "https://github.com/Daemoniorum-LLC/sigil-lang.git", branch: "main"

  # Install from crates.io (published package)
  # For versioned releases, uncomment and update:
  # url "https://github.com/Daemoniorum-LLC/sigil-lang/archive/refs/tags/v0.1.0.tar.gz"
  # sha256 "..."

  depends_on "rust" => :build

  def install
    # Build from HEAD or local source
    if build.head?
      cd "parser" do
        system "cargo", "install", *std_cargo_args
      end
    else
      # Install from crates.io
      system "cargo", "install", "sigil-parser",
             "--root", prefix,
             "--locked"
    end
  end

  def caveats
    <<~EOS
      Sigil is a polysynthetic programming language designed for AI systems.

      Evidence markers:
        !  Known (verified/computed)
        ?  Uncertain (validated)
        ~  Reported (external data)
        ‽  Paradox (self-referential)

      Quick start:
        sigil run hello.sigil    # Run a program
        sigil check hello.sigil  # Type check
        sigil repl               # Interactive REPL

      Documentation: https://github.com/Daemoniorum-LLC/sigil-lang/tree/main/docs
    EOS
  end

  test do
    # Test version
    assert_match "sigil", shell_output("#{bin}/sigil --version")

    # Test basic execution
    (testpath/"hello.sigil").write <<~EOS
      fn main() {
        print("Hello from Sigil!");
      }
    EOS
    assert_match "Hello from Sigil!", shell_output("#{bin}/sigil run #{testpath}/hello.sigil")
  end
end
