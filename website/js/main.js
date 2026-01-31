/**
 * Sigil Language Website
 * Main JavaScript functionality
 */

document.addEventListener('DOMContentLoaded', () => {
  // Privacy-first: Initialize consent before anything else
  initPrivacyConsent();

  // Core functionality
  initTabs();
  initMobileNav();
  initDocsSidebarToggle();
  initSmoothScroll();
  initNavScrollEffect();
  initScrollAnimations();
  initPerfBarAnimations();
  initScrollToTop();
  initKeyboardShortcuts();

  // 11/10 Features
  const phase1Results = initPhase1();
  const phase2Results = initPhase2();
  const phase3Results = initPhase3();

  // Automated self-review (dev mode only)
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    console.group('🔍 11/10 Feature Review');
    console.log('Phase 1:', phase1Results);
    console.log('Phase 2:', phase2Results);
    console.log('Phase 3:', phase3Results);
    console.groupEnd();
  }
});

/**
 * Phase 1: Copy buttons, External link icons, Reading time
 * @returns {Object} Validation results for automated review
 */
function initPhase1() {
  const results = {
    copyButtons: initCopyButtons(),
    externalLinks: initExternalLinkIcons(),
    readingTime: initReadingTime()
  };
  return results;
}

/**
 * Phase 2: Command palette
 * @returns {Object} Validation results
 */
function initPhase2() {
  const results = {
    commandPalette: initCommandPalette()
  };
  return results;
}

/**
 * Phase 3: Scroll spy TOC, Breadcrumbs
 * @returns {Object} Validation results
 */
function initPhase3() {
  const results = {
    scrollSpyTOC: initScrollSpyTOC(),
    breadcrumbs: initBreadcrumbs()
  };
  return results;
}

/**
 * Tab switching for code examples
 */
function initTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const targetTab = button.dataset.tab;

      // Update button states
      tabButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');

      // Update panel visibility
      tabPanels.forEach(panel => {
        panel.classList.remove('active');
        if (panel.id === targetTab) {
          panel.classList.add('active');
        }
      });
    });
  });
}

/**
 * Mobile navigation toggle with proper accessibility
 */
function initMobileNav() {
  const navToggle = document.querySelector('.nav-toggle');
  const navLinks = document.querySelector('.nav-links');

  if (!navToggle || !navLinks) return;

  navToggle.addEventListener('click', () => {
    const isExpanded = navToggle.getAttribute('aria-expanded') === 'true';

    navToggle.setAttribute('aria-expanded', !isExpanded);
    navLinks.classList.toggle('active');
  });

  // Close menu when clicking a link
  navLinks.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      navToggle.setAttribute('aria-expanded', 'false');
      navLinks.classList.remove('active');
    });
  });

  // Close menu when clicking outside
  document.addEventListener('click', (e) => {
    if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
      navToggle.setAttribute('aria-expanded', 'false');
      navLinks.classList.remove('active');
    }
  });

  // Close menu when pressing Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      navToggle.setAttribute('aria-expanded', 'false');
      navLinks.classList.remove('active');
      navToggle.focus();
    }
  });

  // Handle focus trap in mobile nav when open
  navLinks.addEventListener('keydown', (e) => {
    if (e.key === 'Tab' && navLinks.classList.contains('active')) {
      const focusableElements = navLinks.querySelectorAll('a[href]');
      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    }
  });
}

/**
 * Documentation sidebar toggle for mobile
 */
function initDocsSidebarToggle() {
  const sidebar = document.querySelector('.docs-sidebar');
  const sidebarToggle = document.querySelector('.docs-sidebar-toggle');

  if (!sidebar || !sidebarToggle) return;

  sidebarToggle.addEventListener('click', () => {
    const isExpanded = sidebarToggle.getAttribute('aria-expanded') === 'true';
    sidebarToggle.setAttribute('aria-expanded', !isExpanded);
    sidebar.classList.toggle('active');

    // Update button text
    sidebarToggle.querySelector('.toggle-text').textContent =
      isExpanded ? 'Show Menu' : 'Hide Menu';
  });

  // Close sidebar when clicking a link on mobile
  sidebar.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      if (window.innerWidth <= 1024) {
        sidebarToggle.setAttribute('aria-expanded', 'false');
        sidebar.classList.remove('active');
        sidebarToggle.querySelector('.toggle-text').textContent = 'Show Menu';
      }
    });
  });

  // Close sidebar when clicking outside
  document.addEventListener('click', (e) => {
    if (window.innerWidth <= 1024 &&
        !sidebar.contains(e.target) &&
        !sidebarToggle.contains(e.target) &&
        sidebar.classList.contains('active')) {
      sidebarToggle.setAttribute('aria-expanded', 'false');
      sidebar.classList.remove('active');
      sidebarToggle.querySelector('.toggle-text').textContent = 'Show Menu';
    }
  });
}

/**
 * Smooth scroll for anchor links
 */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const href = anchor.getAttribute('href');
      if (href === '#') return;

      e.preventDefault();
      const target = document.querySelector(href);
      if (target) {
        const navHeight = document.querySelector('.nav')?.offsetHeight || 64;
        const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navHeight - 16;

        window.scrollTo({
          top: targetPosition,
          behavior: 'smooth'
        });

        // Update URL without triggering scroll
        history.pushState(null, '', href);
      }
    });
  });
}

/**
 * Navigation background effect on scroll
 * Uses CSS classes instead of inline styles for theme consistency
 */
function initNavScrollEffect() {
  const nav = document.querySelector('.nav');
  if (!nav) return;

  window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    nav.classList.toggle('nav-scrolled', currentScroll > 100);
  }, { passive: true });
}

/**
 * Animate elements on scroll using IntersectionObserver
 */
function initScrollAnimations() {
  const animatedElements = document.querySelectorAll(
    '.evidence-card, .morpheme-card, .agent-layer-group, .backend-card, .install-step, .ai-datum'
  );

  if (animatedElements.length === 0) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, index) => {
      if (entry.isIntersecting) {
        // Stagger the animations slightly
        setTimeout(() => {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }, index * 50);
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -30px 0px'
  });

  animatedElements.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    observer.observe(el);
  });
}

/**
 * Animate performance bars when they come into view
 */
function initPerfBarAnimations() {
  const perfSection = document.querySelector('.perf-comparison');
  if (!perfSection) return;

  const perfFills = perfSection.querySelectorAll('.perf-fill');

  // Store original widths and reset to 0
  const widths = [];
  perfFills.forEach(fill => {
    widths.push(fill.style.width);
    fill.style.width = '0';
  });

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        // Animate bars to their target widths
        perfFills.forEach((fill, index) => {
          setTimeout(() => {
            fill.style.width = widths[index];
          }, index * 200);
        });
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.3
  });

  observer.observe(perfSection);
}

/**
 * Scroll-to-top button for long pages
 */
function initScrollToTop() {
  // Create the button dynamically
  const scrollBtn = document.createElement('button');
  scrollBtn.className = 'scroll-to-top';
  scrollBtn.setAttribute('aria-label', 'Scroll to top');
  scrollBtn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 15l-6-6-6 6"/></svg>`;
  document.body.appendChild(scrollBtn);

  // Show/hide based on scroll position
  const toggleVisibility = () => {
    scrollBtn.classList.toggle('visible', window.pageYOffset > 400);
  };

  window.addEventListener('scroll', toggleVisibility, { passive: true });
  toggleVisibility(); // Check initial state

  // Scroll to top on click
  scrollBtn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

/**
 * Keyboard shortcuts for power users
 * g h = Go Home
 * g d = Go to Docs
 * g l = Go to Learn
 * g a = Go to Agents
 * g g = Go to GitHub
 */
function initKeyboardShortcuts() {
  let keySequence = '';
  let keyTimeout;

  document.addEventListener('keydown', (e) => {
    // Don't trigger shortcuts when typing in inputs
    if (e.target.matches('input, textarea, [contenteditable]')) return;

    // Clear sequence after 500ms
    clearTimeout(keyTimeout);
    keyTimeout = setTimeout(() => { keySequence = ''; }, 500);

    keySequence += e.key.toLowerCase();

    // Handle shortcuts
    const shortcuts = {
      'gh': '/',
      'gd': '/pages/docs.html',
      'gl': '/pages/learn.html',
      'ga': '/pages/agents.html',
      'gg': 'https://github.com/Daemoniorum-LLC/sigil-lang'
    };

    if (shortcuts[keySequence]) {
      e.preventDefault();
      const url = shortcuts[keySequence];
      if (url.startsWith('http')) {
        window.open(url, '_blank', 'noopener,noreferrer');
      } else {
        window.location.href = url;
      }
      keySequence = '';
    }

    // Clear if sequence gets too long
    if (keySequence.length > 2) {
      keySequence = e.key.toLowerCase();
    }
  });
}

/**
 * Reduce motion preference support
 */
if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  document.documentElement.style.setProperty('--transition-fast', '0ms');
  document.documentElement.style.setProperty('--transition-normal', '0ms');
}

/* ============================================
   11/10 FEATURES - Phase 1
   ============================================ */

/**
 * Copy text to clipboard with fallback for older browsers
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>} - Success status
 */
async function copyToClipboard(text) {
  // Modern Clipboard API (most browsers since 2018+)
  if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (err) {
      // Clipboard API failed (permissions, iframe, etc.) - fall through to fallback
      console.warn('Clipboard API failed, trying fallback:', err);
    }
  }

  // Fallback: execCommand for older browsers (Safari <13.1, older Edge, IE11)
  try {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    // Prevent scrolling to bottom of page
    textarea.style.cssText = 'position:fixed;top:0;left:0;width:2em;height:2em;padding:0;border:none;outline:none;box-shadow:none;background:transparent;';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    const success = document.execCommand('copy');
    document.body.removeChild(textarea);

    if (!success) {
      throw new Error('execCommand returned false');
    }
    return true;
  } catch (err) {
    console.error('All copy methods failed:', err);
    return false;
  }
}

// SVG icons for copy button states
const COPY_ICONS = {
  default: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`,
  success: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>`,
  error: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`
};

/**
 * Copy-to-clipboard buttons for code blocks
 * Includes fallback for older browsers and user-visible error states
 * TDD: Returns { count, success } for validation
 */
function initCopyButtons() {
  const codeBlocks = document.querySelectorAll('.code-block, .feature-code, pre code');
  let count = 0;

  codeBlocks.forEach(block => {
    // Skip if already has a copy button
    if (block.parentElement?.querySelector('.copy-btn')) return;

    // Get the container (pre or div)
    const container = block.closest('.code-block, .feature-code') || block.parentElement;
    if (!container || container.querySelector('.copy-btn')) return;

    // Make container position relative for button positioning
    container.style.position = 'relative';

    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-btn';
    copyBtn.setAttribute('aria-label', 'Copy code to clipboard');
    copyBtn.innerHTML = COPY_ICONS.default;

    // Reset button to default state
    function resetButton() {
      copyBtn.classList.remove('copied', 'copy-error');
      copyBtn.innerHTML = COPY_ICONS.default;
      copyBtn.setAttribute('aria-label', 'Copy code to clipboard');
    }

    copyBtn.addEventListener('click', async () => {
      const code = block.textContent || '';

      const success = await copyToClipboard(code);

      if (success) {
        copyBtn.classList.add('copied');
        copyBtn.classList.remove('copy-error');
        copyBtn.innerHTML = COPY_ICONS.success;
        copyBtn.setAttribute('aria-label', 'Copied!');
        // Announce to screen readers
        announceToScreenReader('Code copied to clipboard');
      } else {
        copyBtn.classList.add('copy-error');
        copyBtn.classList.remove('copied');
        copyBtn.innerHTML = COPY_ICONS.error;
        copyBtn.setAttribute('aria-label', 'Failed to copy. Try selecting and copying manually.');
        announceToScreenReader('Failed to copy code. Try selecting and copying manually.');
      }

      setTimeout(resetButton, 2000);
    });

    container.appendChild(copyBtn);
    count++;
  });

  return { count, success: count > 0 };
}

/**
 * Announce message to screen readers via live region
 * @param {string} message - Message to announce
 */
function announceToScreenReader(message) {
  let announcer = document.getElementById('sr-announcer');
  if (!announcer) {
    announcer = document.createElement('div');
    announcer.id = 'sr-announcer';
    announcer.setAttribute('aria-live', 'polite');
    announcer.setAttribute('aria-atomic', 'true');
    announcer.className = 'sr-only';
    document.body.appendChild(announcer);
  }
  // Clear and re-set to trigger announcement
  announcer.textContent = '';
  setTimeout(() => { announcer.textContent = message; }, 100);
}

/**
 * Add external link icons to off-site links
 * TDD: Returns { count, success } for validation
 */
function initExternalLinkIcons() {
  const externalLinks = document.querySelectorAll('a[href^="http"]:not(.nav-github):not(.external-processed)');
  let count = 0;

  externalLinks.forEach(link => {
    // Skip if it's an internal link or already processed
    if (link.hostname === window.location.hostname) return;
    if (link.classList.contains('external-processed')) return;

    link.classList.add('external-link', 'external-processed');

    // Add icon if not already present
    if (!link.querySelector('.external-icon')) {
      const icon = document.createElement('span');
      icon.className = 'external-icon';
      icon.setAttribute('aria-hidden', 'true');
      icon.innerHTML = '↗';
      link.appendChild(icon);
      count++;
    }
  });

  return { count, success: true };
}

/**
 * Add reading time estimates to articles/docs
 * TDD: Returns { pagesProcessed, avgReadingTime } for validation
 */
function initReadingTime() {
  const contentAreas = document.querySelectorAll('.docs-content, .learn-content, main');
  const WORDS_PER_MINUTE = 200;
  let pagesProcessed = 0;
  let totalTime = 0;

  contentAreas.forEach(content => {
    // Skip if already has reading time
    if (document.querySelector('.reading-time')) return;

    const text = content.textContent || '';
    const wordCount = text.trim().split(/\s+/).length;
    const readingTime = Math.ceil(wordCount / WORDS_PER_MINUTE);

    // Only show for substantial content (> 2 min read)
    if (readingTime < 2) return;

    // Find a good place to insert (after h1 or at start)
    const h1 = content.querySelector('h1');
    const insertTarget = h1 || content.firstElementChild;

    if (insertTarget && !insertTarget.parentElement.querySelector('.reading-time')) {
      const badge = document.createElement('p');
      badge.className = 'reading-time';
      badge.innerHTML = `<span class="reading-time-icon">📖</span> ${readingTime} min read`;
      insertTarget.after(badge);
      pagesProcessed++;
      totalTime += readingTime;
    }
  });

  return {
    pagesProcessed,
    avgReadingTime: pagesProcessed ? Math.round(totalTime / pagesProcessed) : 0,
    success: true
  };
}

/* ============================================
   11/10 FEATURES - Phase 2
   ============================================ */

/**
 * Command palette (Cmd+K / Ctrl+K)
 * Includes focus trap for accessibility
 * TDD: Returns { initialized, shortcuts } for validation
 */
function initCommandPalette() {
  // Create palette container
  const palette = document.createElement('div');
  palette.className = 'command-palette';
  palette.setAttribute('role', 'dialog');
  palette.setAttribute('aria-modal', 'true');
  palette.setAttribute('aria-label', 'Command palette');
  palette.innerHTML = `
    <div class="palette-backdrop"></div>
    <div class="palette-container">
      <div class="palette-search">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        <input type="text" class="palette-input" placeholder="Type a command or search..." autocomplete="off" />
        <kbd class="palette-kbd">ESC</kbd>
      </div>
      <ul class="palette-results" role="listbox"></ul>
      <div class="palette-footer">
        <span><kbd>↑↓</kbd> Navigate</span>
        <span><kbd>↵</kbd> Select</span>
        <span><kbd>ESC</kbd> Close</span>
      </div>
    </div>
  `;
  document.body.appendChild(palette);

  const paletteContainer = palette.querySelector('.palette-container');
  const input = palette.querySelector('.palette-input');
  const results = palette.querySelector('.palette-results');
  const backdrop = palette.querySelector('.palette-backdrop');

  // Focus trap state
  let previouslyFocusedElement = null;

  // Command definitions
  const commands = [
    { id: 'home', title: 'Go to Home', keywords: 'home index main', action: () => window.location.href = '/' },
    { id: 'docs', title: 'Go to Documentation', keywords: 'docs documentation reference', action: () => window.location.href = '/pages/docs.html' },
    { id: 'learn', title: 'Go to Learn', keywords: 'learn tutorial guide', action: () => window.location.href = '/pages/learn.html' },
    { id: 'agents', title: 'Go to Agent Infrastructure', keywords: 'agents ai infrastructure', action: () => window.location.href = '/pages/agents.html' },
    { id: 'examples', title: 'Go to Examples', keywords: 'examples code samples', action: () => window.location.href = '/pages/examples.html' },
    { id: 'contribute', title: 'Go to Contributing', keywords: 'contribute contributing help', action: () => window.location.href = '/pages/contributing.html' },
    { id: 'github', title: 'Open GitHub', keywords: 'github repo source code', action: () => window.open('https://github.com/Daemoniorum-LLC/sigil-lang', '_blank') },
    { id: 'top', title: 'Scroll to Top', keywords: 'top scroll up', action: () => window.scrollTo({ top: 0, behavior: 'smooth' }) },
    { id: 'theme', title: 'Toggle Theme (Coming Soon)', keywords: 'theme dark light mode', action: () => {} }
  ];

  let selectedIndex = 0;
  let filteredCommands = [...commands];

  // Get all focusable elements in the palette
  function getFocusableElements() {
    return paletteContainer.querySelectorAll(
      'input, button, [tabindex]:not([tabindex="-1"]), [href], select, textarea'
    );
  }

  // Render results
  function renderResults() {
    results.innerHTML = filteredCommands.map((cmd, i) => `
      <li class="palette-item ${i === selectedIndex ? 'selected' : ''}"
          role="option"
          aria-selected="${i === selectedIndex}"
          data-index="${i}"
          tabindex="-1">
        <span class="palette-item-title">${cmd.title}</span>
      </li>
    `).join('');
  }

  // Filter commands
  function filterCommands(query) {
    const q = query.toLowerCase();
    filteredCommands = commands.filter(cmd =>
      cmd.title.toLowerCase().includes(q) ||
      cmd.keywords.toLowerCase().includes(q)
    );
    selectedIndex = 0;
    renderResults();
  }

  // Open palette
  function openPalette() {
    // Store currently focused element to restore later
    previouslyFocusedElement = document.activeElement;

    // Prevent body scroll while open
    document.body.style.overflow = 'hidden';

    palette.classList.add('open');
    input.value = '';
    filteredCommands = [...commands];
    selectedIndex = 0;
    renderResults();
    input.focus();
  }

  // Close palette
  function closePalette() {
    palette.classList.remove('open');

    // Restore body scroll
    document.body.style.overflow = '';

    // Restore focus to previously focused element
    if (previouslyFocusedElement && typeof previouslyFocusedElement.focus === 'function') {
      previouslyFocusedElement.focus();
    }
    previouslyFocusedElement = null;
  }

  // Execute selected command
  function executeSelected() {
    if (filteredCommands[selectedIndex]) {
      closePalette();
      filteredCommands[selectedIndex].action();
    }
  }

  // Focus trap handler
  function handleFocusTrap(e) {
    if (!palette.classList.contains('open')) return;
    if (e.key !== 'Tab') return;

    const focusableElements = getFocusableElements();
    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    // Shift+Tab on first element -> wrap to last
    if (e.shiftKey && document.activeElement === firstElement) {
      e.preventDefault();
      lastElement.focus();
      return;
    }

    // Tab on last element -> wrap to first
    if (!e.shiftKey && document.activeElement === lastElement) {
      e.preventDefault();
      firstElement.focus();
      return;
    }

    // If somehow focus escaped, bring it back
    if (!paletteContainer.contains(document.activeElement)) {
      e.preventDefault();
      firstElement.focus();
    }
  }

  // Event listeners
  document.addEventListener('keydown', (e) => {
    // Focus trap for open palette
    if (palette.classList.contains('open')) {
      handleFocusTrap(e);
    }

    // Don't trigger Cmd+K in inputs (except palette input)
    if (e.target.matches('input:not(.palette-input), textarea, [contenteditable]')) return;

    // Cmd/Ctrl + K to open
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      palette.classList.contains('open') ? closePalette() : openPalette();
    }
  });

  input.addEventListener('input', (e) => filterCommands(e.target.value));

  input.addEventListener('keydown', (e) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        selectedIndex = (selectedIndex + 1) % filteredCommands.length;
        renderResults();
        break;
      case 'ArrowUp':
        e.preventDefault();
        selectedIndex = (selectedIndex - 1 + filteredCommands.length) % filteredCommands.length;
        renderResults();
        break;
      case 'Enter':
        e.preventDefault();
        executeSelected();
        break;
      case 'Escape':
        closePalette();
        break;
    }
  });

  results.addEventListener('click', (e) => {
    const item = e.target.closest('.palette-item');
    if (item) {
      selectedIndex = parseInt(item.dataset.index, 10);
      executeSelected();
    }
  });

  backdrop.addEventListener('click', closePalette);

  return { initialized: true, shortcuts: commands.length };
}

/* ============================================
   11/10 FEATURES - Phase 3
   ============================================ */

/**
 * Scroll spy TOC for docs pages
 * Creates a floating table of contents that highlights current section
 * Uses IntersectionObserver for robust tracking (works with CSS transforms, sticky headers, dynamic content)
 * TDD: Returns { created, headings } for validation
 */
function initScrollSpyTOC() {
  const docsContent = document.querySelector('.docs-content');
  if (!docsContent) return { created: false, headings: 0 };

  // Find all headings with IDs
  const headings = docsContent.querySelectorAll('h2[id], h3[id]');
  if (headings.length < 3) return { created: false, headings: headings.length };

  // Create TOC container
  const toc = document.createElement('nav');
  toc.className = 'scroll-spy-toc';
  toc.setAttribute('aria-label', 'Table of contents');
  toc.innerHTML = `
    <div class="toc-header">
      <span class="toc-title">On this page</span>
    </div>
    <ul class="toc-list"></ul>
  `;

  const tocList = toc.querySelector('.toc-list');

  // Populate TOC and build heading-to-link map
  const headingToLink = new Map();
  headings.forEach(heading => {
    const li = document.createElement('li');
    li.className = `toc-item toc-${heading.tagName.toLowerCase()}`;
    const link = document.createElement('a');
    link.href = `#${heading.id}`;
    link.className = 'toc-link';
    link.textContent = heading.textContent;
    li.appendChild(link);
    tocList.appendChild(li);
    headingToLink.set(heading, link);
  });

  // Insert TOC after sidebar or at start of docs-layout
  const docsLayout = document.querySelector('.docs-layout');
  if (docsLayout) {
    docsLayout.appendChild(toc);
  }

  // Track visible headings with IntersectionObserver
  const visibleHeadings = new Set();
  let currentActiveId = null;

  function updateActiveLink() {
    // Find the first visible heading, or the last one that was above viewport
    let activeHeading = null;

    // Convert headings to array to maintain document order
    const headingsArray = Array.from(headings);

    // First, check if any heading is currently visible
    for (const heading of headingsArray) {
      if (visibleHeadings.has(heading.id)) {
        activeHeading = heading;
        break;
      }
    }

    // If no heading is visible, find the last heading that's above viewport
    if (!activeHeading) {
      for (let i = headingsArray.length - 1; i >= 0; i--) {
        const heading = headingsArray[i];
        const rect = heading.getBoundingClientRect();
        if (rect.top < 150) {
          activeHeading = heading;
          break;
        }
      }
    }

    // Default to first heading if nothing else applies
    if (!activeHeading && headingsArray.length > 0) {
      activeHeading = headingsArray[0];
    }

    const newActiveId = activeHeading?.id || null;

    // Only update DOM if active changed
    if (newActiveId !== currentActiveId) {
      headingToLink.forEach((link) => {
        link.classList.remove('active');
      });

      if (activeHeading && headingToLink.has(activeHeading)) {
        headingToLink.get(activeHeading).classList.add('active');
      }

      currentActiveId = newActiveId;
    }
  }

  // Use IntersectionObserver for robust scroll detection
  // rootMargin: negative top margin to trigger when heading is near top of viewport
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          visibleHeadings.add(entry.target.id);
        } else {
          visibleHeadings.delete(entry.target.id);
        }
      });
      updateActiveLink();
    },
    {
      // Observe within the viewport, with offset for header
      rootMargin: '-80px 0px -60% 0px',
      threshold: 0
    }
  );

  // Observe all headings
  headings.forEach(heading => observer.observe(heading));

  // Initial update
  updateActiveLink();

  // Cleanup function (store on element for potential future use)
  toc._cleanup = () => {
    observer.disconnect();
  };

  return { created: true, headings: headings.length };
}

/**
 * Get page name from document metadata
 * Priority: data-breadcrumb-title meta > title before separator > filename
 * @returns {string|null} Page name or null
 */
function getPageNameFromMeta() {
  // 1. Check for explicit breadcrumb meta tag
  const breadcrumbMeta = document.querySelector('meta[name="breadcrumb-title"]');
  if (breadcrumbMeta?.content) {
    return breadcrumbMeta.content;
  }

  // 2. Parse document title (format: "Page Name - Site Name" or "Page Name | Site Name")
  const title = document.title;
  if (title) {
    // Split by common separators and take the first part
    const separators = [' - ', ' | ', ' — ', ' · '];
    for (const sep of separators) {
      if (title.includes(sep)) {
        const pageName = title.split(sep)[0].trim();
        if (pageName && pageName !== 'Sigil') {
          return pageName;
        }
      }
    }
    // No separator found, use full title if it's not just the site name
    if (title !== 'Sigil' && title !== 'Sigil Programming Language') {
      return title;
    }
  }

  // 3. Fall back to h1 content
  const h1 = document.querySelector('h1');
  if (h1?.textContent) {
    return h1.textContent.trim();
  }

  return null;
}

/**
 * Convert filename to readable title
 * e.g., "my-page.html" -> "My Page"
 * @param {string} filename
 * @returns {string}
 */
function filenameToTitle(filename) {
  return filename
    .replace(/\.html?$/i, '')
    .replace(/[-_]/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

/**
 * Breadcrumb navigation
 * Shows current page path: Home > Section > Page
 * Reads page name from document metadata for accuracy
 * TDD: Returns { created, depth } for validation
 */
function initBreadcrumbs() {
  // Only add to subpages
  const path = window.location.pathname;
  if (path === '/' || path === '/index.html') {
    return { created: false, depth: 0 };
  }

  // Skip if breadcrumbs already exist
  if (document.querySelector('.breadcrumbs')) {
    return { created: false, depth: 0, reason: 'already exists' };
  }

  // Parse path into crumbs
  const crumbs = [{ name: 'Home', url: '/' }];

  // Segments to skip (container directories that aren't meaningful)
  const skipSegments = new Set(['pages', 'docs', 'content']);

  const segments = path.split('/').filter(Boolean);
  const pageSegment = segments[segments.length - 1];

  // Add intermediate segments (directories)
  let currentPath = '';
  for (let i = 0; i < segments.length - 1; i++) {
    const segment = segments[i];
    currentPath += '/' + segment;
    if (!skipSegments.has(segment)) {
      crumbs.push({ name: filenameToTitle(segment), url: currentPath });
    }
  }

  // Add current page using metadata
  if (pageSegment) {
    currentPath += '/' + pageSegment;
    const pageName = getPageNameFromMeta() || filenameToTitle(pageSegment);
    crumbs.push({ name: pageName, url: currentPath });
  }

  if (crumbs.length < 2) return { created: false, depth: 0 };

  // Create breadcrumb element
  const nav = document.createElement('nav');
  nav.className = 'breadcrumbs';
  nav.setAttribute('aria-label', 'Breadcrumb');
  nav.innerHTML = `
    <ol class="breadcrumb-list">
      ${crumbs.map((crumb, i) => `
        <li class="breadcrumb-item">
          ${i === crumbs.length - 1
            ? `<span class="breadcrumb-current" aria-current="page">${crumb.name}</span>`
            : `<a href="${crumb.url}" class="breadcrumb-link">${crumb.name}</a>`
          }
        </li>
      `).join('<li class="breadcrumb-separator" aria-hidden="true">/</li>')}
    </ol>
  `;

  // Insert after nav, before main content
  const main = document.querySelector('main') || document.querySelector('.docs-layout');
  if (main) {
    main.insertBefore(nav, main.firstChild);
  }

  return { created: true, depth: crumbs.length };
}

/* ============================================
   PRIVACY CONSENT - GDPR/CCPA Compliant
   Philosophy: Privacy-first, respect DNT, clear choices
   ============================================ */

const CONSENT_KEY = 'sigil_analytics_consent';
const CONSENT_TIMESTAMP_KEY = 'sigil_consent_timestamp';
const GA_MEASUREMENT_ID = 'G-J7K2SF002F';

/**
 * Check if Do Not Track is enabled
 */
function isDNTEnabled() {
  if (typeof navigator === 'undefined') return false;
  return navigator.doNotTrack === '1' ||
         navigator.doNotTrack === 'yes' ||
         window.doNotTrack === '1';
}

/**
 * Get current consent state
 * Returns: 'granted' | 'denied' | 'pending'
 */
function getConsentState() {
  // Always respect Do Not Track
  if (isDNTEnabled()) return 'denied';

  try {
    const consent = localStorage.getItem(CONSENT_KEY);
    const timestamp = localStorage.getItem(CONSENT_TIMESTAMP_KEY);

    if (consent && timestamp) {
      // Consent expires after 12 months (GDPR requirement)
      const twelveMonths = 365 * 24 * 60 * 60 * 1000;
      if (Date.now() - parseInt(timestamp, 10) > twelveMonths) {
        localStorage.removeItem(CONSENT_KEY);
        localStorage.removeItem(CONSENT_TIMESTAMP_KEY);
        return 'pending';
      }
      return consent;
    }
    return 'pending';
  } catch {
    return 'pending';
  }
}

/**
 * Set consent state and update GA
 */
function setConsentState(granted) {
  const status = granted ? 'granted' : 'denied';

  try {
    localStorage.setItem(CONSENT_KEY, status);
    localStorage.setItem(CONSENT_TIMESTAMP_KEY, Date.now().toString());

    // Update GA consent mode
    if (window.gtag) {
      window.gtag('consent', 'update', {
        analytics_storage: status
      });
    }

    // Initialize GA if consent granted
    if (granted && !isDNTEnabled()) {
      initGA4();
    }
  } catch (e) {
    console.warn('[Privacy] Failed to save consent:', e);
  }
}

/**
 * Initialize GA4 with privacy-preserving settings
 */
function initGA4() {
  if (typeof window.gtag !== 'function') return;

  window.gtag('config', GA_MEASUREMENT_ID, {
    anonymize_ip: true,
    allow_google_signals: false,
    allow_ad_personalization_signals: false
  });
}

/**
 * Initialize privacy consent banner
 */
function initPrivacyConsent() {
  const consentState = getConsentState();

  // If consent already decided, just initialize GA if granted
  if (consentState === 'granted') {
    initGA4();
    return;
  }

  // If DNT enabled or already denied, don't show banner
  if (consentState === 'denied') {
    return;
  }

  // Show consent banner after slight delay to avoid flash
  setTimeout(() => {
    showConsentBanner();
  }, 500);
}

/**
 * Create and show the consent banner
 */
function showConsentBanner() {
  const banner = document.createElement('div');
  banner.className = 'privacy-banner';
  banner.setAttribute('role', 'dialog');
  banner.setAttribute('aria-labelledby', 'privacy-title');
  banner.setAttribute('aria-describedby', 'privacy-desc');

  banner.innerHTML = `
    <div class="privacy-content">
      <div class="privacy-text">
        <h2 id="privacy-title" class="privacy-heading">Your Privacy Matters</h2>
        <p id="privacy-desc" class="privacy-body">
          We use analytics to understand how visitors interact with our site.
          This helps us improve your experience. We never sell your data.
          <a href="/pages/privacy.html" class="privacy-link">Privacy Policy</a>
        </p>
      </div>
      <div class="privacy-actions">
        <button class="privacy-btn privacy-accept">Accept Analytics</button>
        <button class="privacy-btn privacy-decline">Decline</button>
      </div>
    </div>
  `;

  document.body.appendChild(banner);

  // Animate in
  requestAnimationFrame(() => {
    banner.classList.add('visible');
  });

  // Handle accept
  banner.querySelector('.privacy-accept').addEventListener('click', () => {
    setConsentState(true);
    hideBanner(banner);
  });

  // Handle decline
  banner.querySelector('.privacy-decline').addEventListener('click', () => {
    setConsentState(false);
    hideBanner(banner);
  });
}

/**
 * Hide and remove the consent banner
 */
function hideBanner(banner) {
  banner.classList.remove('visible');
  setTimeout(() => banner.remove(), 300);
}
