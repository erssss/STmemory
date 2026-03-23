/**
 * Docusaurus Remark plugin to automatically add 'overview' suffix to directory links
 * 
 * This plugin transforms links ending with '/' to end with '/overview'
 * Example: [Link](../api/) -> [Link](../api/overview)
 * 
 * It only processes relative paths (starting with ./ or ../) or absolute paths (starting with /)
 * that end with a slash, and adds 'overview' before the trailing slash.
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
function remarkAutoOverview() {
  return (tree) => {
    visit(tree, 'link', (node) => {
      if (node.url && typeof node.url === 'string') {
        // Check if the link is a relative or absolute path ending with '/'
        // Match patterns like:
        // - ../api/
        // - ./examples/
        // - /docs/api/
        // - ../architecture/
        const isRelativePath = /^\.\.?\/[^)]+\/$/.test(node.url);
        const isAbsolutePath = /^\/[^)]+\/$/.test(node.url);
        
        if (isRelativePath || isAbsolutePath) {
          // Add 'overview' suffix before the trailing '/'
          // ../api/ -> ../api/overview
          // /docs/api/ -> /docs/api/overview
          node.url = node.url.replace(/\/$/, '/overview');
        }
      }
    });
  };
}

module.exports = remarkAutoOverview;

