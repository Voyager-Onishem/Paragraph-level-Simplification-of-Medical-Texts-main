# PowerShell script to run the medical text processing pipeline

# Default values that can be overridden
param (
    [string]$InputFile = "scraped_data/data.json",
    [string]$OutputDir = "scraped_data",
    [int]$MinAbstractLength = 500,
    [double]$MinLengthRatio = 0.15,
    [double]$MaxLengthRatio = 2.0,
    [double]$MinComplexityDiff = 5.0,
    [double]$MinTermPreservation = 0.4,
    [int]$MinParagraphs = 2,
    [int]$MaxParagraphs = 7,
    [double]$MinCompressionRatio = 0.3,
    [double]$MaxCompressionRatio = 0.7,
    [int]$MaxTokenLength = 1024,
    [switch]$SkipComplexity = $false,
    [switch]$SkipTermPreservation = $false,
    [switch]$SkipParagraphStructure = $false,
    [switch]$SkipLengthGuidance = $false,
    [switch]$ForceRepair = $false,
    [switch]$Help = $false
)

# Show help information if requested
if ($Help) {
    Write-Host "Usage: .\process_data.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -InputFile <path>             Path to input data.json file [default: scraped_data/data.json]"
    Write-Host "  -OutputDir <dir>              Directory for output files [default: scraped_data]"
    Write-Host "  -MinAbstractLength <int>      Minimum abstract length [default: 500]"
    Write-Host "  -MinLengthRatio <float>       Minimum PLS/abstract length ratio [default: 0.15]"
    Write-Host "  -MaxLengthRatio <float>       Maximum PLS/abstract length ratio [default: 2.0]"
    Write-Host "  -MinComplexityDiff <float>    Minimum reading ease difference [default: 5.0]"
    Write-Host "  -MinTermPreservation <float>  Minimum medical term preservation [default: 0.4]"
    Write-Host "  -MinParagraphs <int>          Minimum paragraphs for good structure [default: 2]"
    Write-Host "  -MaxParagraphs <int>          Maximum paragraphs for good structure [default: 7]"
    Write-Host "  -MinCompressionRatio <float>  Minimum target/source ratio [default: 0.3]"
    Write-Host "  -MaxCompressionRatio <float>  Maximum target/source ratio [default: 0.7]"
    Write-Host "  -MaxTokenLength <int>         Maximum token length [default: 1024]"
    Write-Host "  -SkipComplexity               Skip complexity filtering"
    Write-Host "  -SkipTermPreservation         Skip term preservation filtering"
    Write-Host "  -SkipParagraphStructure       Skip paragraph structure filtering"
    Write-Host "  -SkipLengthGuidance           Skip length guidance filtering"
    Write-Host "  -ForceRepair                  Force repair of data.json even if it exists"
    Write-Host "  -Help                         Show this help message"
    exit 0
}

# Build the command-line arguments
$args = @("prepare_data\process.py")
$args += "--input_file", $InputFile
$args += "--output_dir", $OutputDir
$args += "--min_abstract_length", $MinAbstractLength
$args += "--min_length_ratio", $MinLengthRatio
$args += "--max_length_ratio", $MaxLengthRatio
$args += "--min_complexity_diff", $MinComplexityDiff
$args += "--min_term_preservation", $MinTermPreservation
$args += "--min_paragraphs", $MinParagraphs
$args += "--max_paragraphs", $MaxParagraphs
$args += "--min_compression_ratio", $MinCompressionRatio
$args += "--max_compression_ratio", $MaxCompressionRatio
$args += "--max_token_length", $MaxTokenLength

# Add optional flags if specified
if ($SkipComplexity) { $args += "--skip_complexity" }
if ($SkipTermPreservation) { $args += "--skip_term_preservation" }
if ($SkipParagraphStructure) { $args += "--skip_paragraph_structure" }
if ($SkipLengthGuidance) { $args += "--skip_length_guidance" }
if ($ForceRepair) { $args += "--force_repair" }

# Display the command being executed
Write-Host "Running: python $($args -join ' ')" -ForegroundColor Cyan

# Execute the Python script with the arguments
try {
    & python $args
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Processing completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Processing failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "An error occurred while running the script:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# Example usage information
Write-Host ""
Write-Host "Example usage:" -ForegroundColor Yellow
Write-Host "  .\process_data.ps1 -MinAbstractLength 300 -MinComplexityDiff 3.0" -ForegroundColor Yellow
Write-Host "  .\process_data.ps1 -SkipTermPreservation -SkipParagraphStructure" -ForegroundColor Yellow
Write-Host "  .\process_data.ps1 -InputFile 'data/custom_data.json' -OutputDir 'output'" -ForegroundColor Yellow
