# Sigil Website (DEPRECATED)

> **This static HTML website is deprecated.**
>
> The canonical Sigil website is now **website-qliphoth**, located at:
> `/home/crook/dev2/workspace/sigil/website-qliphoth/`
>
> The canonical site is written entirely in Sigil and compiles to WebAssembly.

## Why Deprecated?

This directory contains the original static HTML/CSS/JS website. It has been superseded by **website-qliphoth**, which:

1. Is written entirely in Sigil using the Qliphoth UI framework
2. Compiles to WebAssembly (6.3 KB)
3. Demonstrates Sigil's web capabilities
4. Is the source deployed to [sigil-lang.com](https://sigil-lang.com)

## Fallback Use

This static site is kept as a fallback for browsers that don't support WASM. The Qliphoth site references it at `../website/index.html` if WASM loading fails.

## Do Not Update

Please make all website changes to **website-qliphoth** instead. This directory should not receive new features or content updates.

## Canonical Site Location

```
/home/crook/dev2/workspace/sigil/website-qliphoth/
├── src/
│   ├── main.sigil       # Main website source
│   ├── pages.sigil      # Page definitions
│   └── components.sigil # Reusable components
└── dist/
    ├── index.html       # HTML shell
    ├── site.wasm        # Compiled WASM
    └── sigil_runtime.js # JS runtime
```
