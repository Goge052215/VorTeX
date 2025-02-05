#!/bin/bash

if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

if ! command -v tex &> /dev/null; then
    echo "Installing BasicTeX..."
    brew install --cask basictex
    sleep 5
    eval "$(/usr/libexec/path_helper)"
    export PATH="/Library/TeX/texbin:$PATH"
else
    echo "BasicTeX is already installed."
fi

if ! command -v tlmgr &> /dev/null; then
    echo "Error: tlmgr not found. Please ensure BasicTeX was installed correctly."
    exit 1
fi

echo "Updating TeX Live Manager..."
sudo tlmgr update --self

echo "Installing required LaTeX packages..."
sudo tlmgr install standalone
sudo tlmgr install babel-english
sudo tlmgr install amsmath
sudo tlmgr install amssymb
sudo tlmgr install preview
sudo tlmgr install dvipng
sudo tlmgr install dvisvgm
sudo tlmgr install geometry
sudo tlmgr install graphics
sudo tlmgr install tools

echo "LaTeX package installation complete."

echo "Verifying installations..."
command -v dvisvgm >/dev/null 2>&1 || { echo "Warning: dvisvgm is not properly installed"; }
command -v dvipng >/dev/null 2>&1 || { echo "Warning: dvipng is not properly installed"; }
