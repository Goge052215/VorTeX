#!/bin/bash

# Function to check if Monaspace Neon font is installed
check_font_installed() {
    # Check common font locations
    if [ -f ~/Library/Fonts/MonaspaceNeon-Regular.otf ] || \
       [ -f /Library/Fonts/MonaspaceNeon-Regular.otf ] || \
       [ -f /System/Library/Fonts/MonaspaceNeon-Regular.otf ]; then
        return 0  # Font is installed
    fi
    
    # Check if font is in the system using fc-list (if available)
    if command -v fc-list &> /dev/null; then
        if fc-list | grep -i "monaspace neon" &> /dev/null; then
            return 0  # Font is installed
        fi
    fi
    
    return 1  # Font is not installed
}

if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
fi

# Check if font is already installed
if check_font_installed; then
    echo "Monaspace Neon font is already installed."
else
    echo "Installing the Monaspace Neon font..."
    brew tap homebrew/cask-fonts || true
    brew install --cask font-monaspace || true
    echo "Font installation complete."
fi
