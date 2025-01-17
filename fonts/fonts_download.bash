#!/bin/bash

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
fi

# Install font using the new tap
echo "Installing the Monaspace Neon font..."
brew tap homebrew/cask-fonts || true  # Continue even if tap exists
brew install --cask font-monaspace || true  # Install font, continue if already installed

echo "Font installation complete."