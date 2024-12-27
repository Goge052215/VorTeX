$repoUrl = "https://github.com/githubnext/monaspace.git"
$localPath = "$env:TEMP\monaspace"

if (-Not (Test-Path $localPath)) {
    Write-Host "Cloning Monaspace repository..."
    git clone $repoUrl $localPath
} else {
    Write-Host "Monaspace repository already cloned."
}

$fontDirs = @("$localPath\fonts\otf", "$localPath\fonts\variable", "$localPath\fonts\frozen")

function Install-Font {
    param (
        [string]$fontPath
    )
    Write-Host "Installing font: $fontPath"
    Copy-Item -Path $fontPath -Destination "C:\Windows\Fonts" -Force
}

foreach ($dir in $fontDirs) {
    if (Test-Path $dir) {
        Get-ChildItem -Path $dir -Filter *.ttf | ForEach-Object {
            Install-Font -fontPath $_.FullName
        }
    } else {
        Write-Host "Directory not found: $dir"
    }
}

Write-Host "Font installation complete."