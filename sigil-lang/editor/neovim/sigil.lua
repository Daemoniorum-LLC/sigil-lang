-- Sigil Language Configuration for Neovim
--
-- Option 1: Add to your init.lua or a plugin file
-- Option 2: Save as ~/.config/nvim/after/plugin/sigil.lua

-- Register the filetype
vim.filetype.add({
  extension = {
    sigil = "sigil",
    sg = "sigil",
  },
})

-- Basic syntax highlighting (until tree-sitter-sigil exists)
vim.api.nvim_create_autocmd("FileType", {
  pattern = "sigil",
  callback = function()
    -- Use Rust syntax as a reasonable fallback
    vim.bo.syntax = "rust"
    -- Set comment strings
    vim.bo.commentstring = "// %s"
    -- Indentation
    vim.bo.tabstop = 4
    vim.bo.shiftwidth = 4
    vim.bo.expandtab = true
  end,
})

-- LSP Configuration (requires nvim-lspconfig)
local ok, lspconfig = pcall(require, "lspconfig")
if ok then
  local configs = require("lspconfig.configs")

  -- Register sigil-oracle if not already registered
  if not configs.sigil_oracle then
    configs.sigil_oracle = {
      default_config = {
        cmd = { "sigil-oracle" },
        filetypes = { "sigil" },
        root_dir = function(fname)
          return lspconfig.util.find_git_ancestor(fname)
            or lspconfig.util.path.dirname(fname)
        end,
        settings = {},
      },
      docs = {
        description = [[
Oracle - Language Server for Sigil

https://github.com/Daemoniorum-LLC/sigil-lang/tree/main/tools/oracle
]],
      },
    }
  end

  -- Set up the server
  lspconfig.sigil_oracle.setup({
    -- Add your on_attach and capabilities here
    -- on_attach = your_on_attach_function,
    -- capabilities = your_capabilities,
  })
end
