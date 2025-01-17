#!/bin/bash

if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
fi

echo "Installing the Monaspace Neon font..."
brew tap homebrew/cask-fonts || true
brew install --cask font-monaspace || true
echo "Font installation complete."
