/**
 * Docusaurus Remark plugin to normalize document links
 * 
 * This plugin transforms relative document links to Docusaurus-compatible format.
 * It handles cases where links use relative paths like ../api/memory.md and converts them
 * to absolute paths that Docusaurus can correctly resolve.
 * 
 * Examples:
 * - [Link](../api/memory) -> [Link](/docs/api/memory)
 * - [Link](../api/memory.md) -> [Link](/docs/api/memory)
 * - [Link](./guides/configuration) -> [Link](/docs/guides/configuration)
 * - [Link](./guides/configuration.md) -> [Link](/docs/guides/configuration)
 * 
 * It only processes relative paths (starting with ./ or ../) that reference
 * local documentation files, and converts them to absolute paths starting with /docs/.
 */

/**
 * Simple visitor function to traverse the AST
 */
function visit(tree, type, visitor) {
  if (!tree || typeof tree !== 'object') {
    return;
  }

  if (Array.isArray(tree)) {
    for (const node of tree) {
      visit(node, type, visitor);
    }
    return;
  }

  if (tree.type === type) {
    visitor(tree);
  }

  // Recursively visit children
  for (const key in tree) {
    if (key !== 'type' && key !== 'position' && typeof tree[key] === 'object') {
      visit(tree[key], type, visitor);
    }
  }
}

/**
 * Normalize a relative path to Docusaurus absolute path format
 * @param {string} url - The URL to normalize
 * @param {string} currentPath - The current file path (optional, for context)
 * @returns {string} - The normalized URL (absolute path starting with /docs/)
 */
function normalizeDocLink(url, currentPath) {
  // Skip external links, anchors, and mailto links
  if (url.includes('://') || url.startsWith('#') || url.startsWith('mailto:')) {
    return url;
  }

  // Skip if already an absolute path starting with /docs/
  if (url.startsWith('/docs/')) {
    return url;
  }

  // Process relative paths (./ or ../)
  const relativePathMatch = /^(\.\.?\/)(.+)$/.test(url);
  if (!relativePathMatch) {
    return url;
  }

  // Remove leading ./ or ../
  let normalized = url.replace(/^\.\.?\/+/, '');
  
  // Remove .md suffix if present
  normalized = normalized.replace(/\.md$/, '');
  
  // Remove trailing slash
  normalized = normalized.replace(/\/$/, '');
  
  // Convert to absolute path starting with /docs/
  return '/docs/' + normalized;
}

/**
 * @param {import('unified').Plugin} options
 */
function remarkNormalizeDocLinks() {
  return (tree, file) => {
    const currentPath = file?.path || '';
    
    visit(tree, 'link', (node) => {
      if (node.url && typeof node.url === 'string') {
        const normalized = normalizeDocLink(node.url, currentPath);
        if (normalized !== node.url) {
          node.url = normalized;
        }
      }
    });
  };
}

module.exports = remarkNormalizeDocLinks;

