/**
 * Docusaurus Remark plugin to automatically remove '.md' suffix from links
 * 
 * This plugin transforms links ending with '.md' to remove the suffix
 * Example: [Link](../api/memory.md) -> [Link](../api/memory)
 * 
 * It only processes relative paths (starting with ./ or ../) or absolute paths (starting with /)
 * that end with '.md', and removes the '.md' suffix.
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
 * @param {import('unified').Plugin} options
 */
function remarkRemoveMdSuffix() {
  return (tree) => {
    visit(tree, 'link', (node) => {
      if (node.url && typeof node.url === 'string') {
        // Only process relative paths (./ or ../) or absolute paths (/)
        // Skip external links (http://, https://, mailto:, etc.)
        const isRelativePath = /^\.\.?\/.*\.md$/.test(node.url);
        const isAbsolutePath = /^\/.*\.md$/.test(node.url);
        
        if (isRelativePath || isAbsolutePath) {
          // Remove '.md' suffix
          // ../api/memory.md -> ../api/memory
          // /docs/api/memory.md -> /docs/api/memory
          node.url = node.url.replace(/\.md$/, '');
        }
      }
    });
  };
}

module.exports = remarkRemoveMdSuffix;

