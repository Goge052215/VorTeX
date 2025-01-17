$repoUrl = "https://github.com/githubnext/monaspace.git"
$localPath = "$env:TEMP\monaspace"

try {
    if (!(Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "Git is not installed. Please install Git first."
        exit 1
    }

    if (-Not (Test-Path $localPath)) {
        Write-Host "Cloning Monaspace repository..."
        git clone $repoUrl $localPath
        if (-Not $?) {
            throw "Failed to clone repository"
        }
    } else {
        Write-Host "Updating Monaspace repository..."
        Set-Location $localPath
        git pull
        if (-Not $?) {
            throw "Failed to update repository"
        }
    }

    $fontDirs = @(
        "$localPath\fonts\otf",
        "$localPath\fonts\variable",
        "$localPath\fonts\frozen"
    )

    function Install-Font {
        param (
            [string]$fontPath
        )
        try {
            Write-Host "Installing font: $fontPath"
            Copy-Item -Path $fontPath -Destination "C:\Windows\Fonts" -Force
            if (-Not $?) {
                throw "Failed to copy font file"
            }
        }
        catch {
            Write-Host "Error installing font $fontPath : $_" -ForegroundColor Red
        }
    }

    foreach ($dir in $fontDirs) {
        if (Test-Path $dir) {
            Get-ChildItem -Path $dir -Filter "*.ttf" | ForEach-Object {
                Install-Font -fontPath $_.FullName
            }
            Get-ChildItem -Path $dir -Filter "*.otf" | ForEach-Object {
                Install-Font -fontPath $_.FullName
            }
        } else {
            Write-Host "Directory not found: $dir" -ForegroundColor Yellow
        }
    }

    Write-Host "Font installation complete." -ForegroundColor Green

} catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
    exit 1
} finally {
    if (Test-Path $localPath) {
        Remove-Item -Path $localPath -Recurse -Force -ErrorAction SilentlyContinue
    }
}