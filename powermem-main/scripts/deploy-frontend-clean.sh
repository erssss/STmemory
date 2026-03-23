#!/bin/bash
# Frontend deployment script for PowerMem documentation
# This script follows the same steps as the GitHub Actions workflow

set -e

echo "=========================================="
echo "PowerMem Frontend Deployment Script"
echo "=========================================="
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    echo "Please install Node.js 20 or higher"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 20 ]; then
    echo "Error: Node.js version must be 20 or higher (current: $(node -v))"
    exit 1
fi

echo "✓ Node.js version: $(node -v)"
echo ""

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Step 1: Clean cache
echo "Step 1: Cleaning cache..."
cd docs/website
rm -rf .docusaurus node_modules build || true
echo "✓ Cache cleaned"
echo ""

# Step 2: Sync documentation files
echo "Step 2: Syncing documentation files..."
cd "$PROJECT_ROOT"

# Delete all files in docs/website/docs directory
echo "  Cleaning docs/website/docs directory..."
rm -rf docs/website/docs/*
find docs/website/docs -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} + 2>/dev/null || true
