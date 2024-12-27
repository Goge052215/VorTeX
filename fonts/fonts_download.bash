#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Homebrew is installed
if ! command_exists brew; then
    echo "Homebrew is not installed. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
fi

# Tap the Homebrew fonts repository if not already tapped
if ! brew tap | grep -q "homebrew/cask-fonts"; then
    echo "Tapping the Homebrew fonts repository..."
    brew tap homebrew/cask-fonts
else
    echo "Homebrew fonts repository is already tapped."
fi

# Install the Monaspace Neon font
echo "Installing the Monaspace Neon font..."
brew install --cask font-monaspace

echo "Font installation complete."