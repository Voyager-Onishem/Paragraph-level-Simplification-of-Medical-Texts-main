# Basic parameters
$GENERATION_FILE = "trained_models/bart-ul_both/best_model/generation/test_generations.json"
$OUTPUT_DIR = "evaluation_results/bart-ul_both"

# Sampling parameters
$SAMPLE_SIZE = 0  # 0 means evaluate all examples
$RANDOM_SEED = 42  # Fixed seed for reproducibility

# Evaluation options
$SKIP_FACTUAL = $false  # Set to $true to skip factual consistency evaluation
$SKIP_BERTSCORE = $false  # Set to $true to skip BERTScore calculation
$MODEL_NAME = "BART-UL-Both"  # Name for labeling in results

# Construct base command
$cmd = "python evaluation/evaluate.py " + `
    "--generation_file `"$GENERATION_FILE`" " + `
    "--output_dir `"$OUTPUT_DIR`" "

# Add sampling parameters if needed
if ($SAMPLE_SIZE -gt 0) {
    $cmd += "--sample_size $SAMPLE_SIZE " + `
           "--random_seed $RANDOM_SEED "
}

# Add evaluation options
if ($SKIP_FACTUAL) {
    $cmd += "--skip_factual "
}

if ($SKIP_BERTSCORE) {
    $cmd += "--skip_bertscore "
}

if ($MODEL_NAME) {
    $cmd += "--model_name `"$MODEL_NAME`" "
}

# Display info before running
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Medical Text Simplification Evaluation" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Generation file: $GENERATION_FILE"
Write-Host "Output directory: $OUTPUT_DIR"
if ($SAMPLE_SIZE -gt 0) { Write-Host "Sample size: $SAMPLE_SIZE" }
Write-Host "Model name: $MODEL_NAME"
Write-Host "Skipping factual evaluation: $SKIP_FACTUAL"
Write-Host "Skipping BERTScore: $SKIP_BERTSCORE"
Write-Host "===============================================" -ForegroundColor Cyan

# Track execution time
$startTime = Get-Date
Write-Host "Starting evaluation at $startTime" -ForegroundColor Yellow

# Execute command
Write-Host "Executing command: $cmd" -ForegroundColor Gray
Invoke-Expression $cmd

# Show completion info
$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Evaluation completed at $endTime" -ForegroundColor Green
Write-Host "Total execution time: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host "Results saved to: $OUTPUT_DIR" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan