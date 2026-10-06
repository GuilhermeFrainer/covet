param (
    [switch]$FullYelp = $false,
    [switch]$SkipYelpConvert = $false
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting Data Representation Generation Pipeline" -ForegroundColor Cyan
Write-Host "Default: Dual Preprocessing (Unstemmed & Stemmed+Stopword-Removed)" -ForegroundColor Cyan
if ($FullYelp) {
    Write-Host "Mode: Full Yelp Dataset Enabled" -ForegroundColor Yellow
} else {
    Write-Host "Mode: Sampled Yelp (10k) Enabled" -ForegroundColor Green
}
if ($SkipYelpConvert) {
    Write-Host "Option: Skipping Yelp JSON-to-Parquet conversion" -ForegroundColor Yellow
}
Write-Host "==========================================================" -ForegroundColor Cyan

# Determine datasets to process
if ($FullYelp) {
    $datasets = @("fed", "anes", "gadarian", "trump", "trump_s25000", "yelp")
} else {
    $datasets = @("fed", "anes", "gadarian", "trump", "trump_s25000", "yelp_s10000")
}

foreach ($dataset in $datasets) {
    Write-Host "`n----------------------------------------------------------" -ForegroundColor Yellow
    Write-Host " Processing dataset: $dataset" -ForegroundColor Yellow
    Write-Host "----------------------------------------------------------" -ForegroundColor Yellow

    # Sampled Trump reuses trump's preprocessing and embeddings (see
    # docs/trump_downsampling.md), so only the sample and R objects are built.
    if ($dataset -eq "trump_s25000") {
        Write-Host "[1/4] Sampling 25k documents from processed Trump embeddings..." -ForegroundColor Gray
        uv run python scripts/data_prep/sample_trump.py --n 25000
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to sample Trump 25k"
            exit $LASTEXITCODE
        }

        Write-Host "[2/4] Tokenizing texts for the STM bag-of-words..." -ForegroundColor Gray
        uv run python scripts/data_prep/build_bow_tokens.py --dataset $dataset
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to tokenize $dataset"
            exit $LASTEXITCODE
        }

        Write-Host "[3/4] Building R BoW and STM representations (Unstemmed)..." -ForegroundColor Gray
        Rscript scripts/r_scripts/build_bow.R --dataset $dataset --text_col clean_text
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to build unstemmed BoW for $dataset"
            exit $LASTEXITCODE
        }

        Write-Host "[4/4] Building R BoW and STM representations (Stemmed)..." -ForegroundColor Gray
        Rscript scripts/r_scripts/build_bow.R --dataset $dataset --text_col clean_text_stemmed --output_suffix _stemmed
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to build stemmed BoW for $dataset"
            exit $LASTEXITCODE
        }
        continue
    }

    # 1. Build dataset
    if ($dataset -eq "yelp_s10000") {
        Write-Host "[1/6] Building Yelp base dataset and early 10k sample..." -ForegroundColor Gray
        if ($SkipYelpConvert) {
            uv run scripts/data_prep/build_datasets.py --dataset yelp --skip-convert
        } else {
            uv run scripts/data_prep/build_datasets.py --dataset yelp
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to build Yelp base dataset"
            exit $LASTEXITCODE
        }
        uv run python scripts/data_prep/sample_yelp_interim.py
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to sample Yelp 10k"
            exit $LASTEXITCODE
        }
    } else {
        Write-Host "[1/6] Building dataset: $dataset..." -ForegroundColor Gray
        if ($dataset -eq "yelp" -and $SkipYelpConvert) {
            uv run scripts/data_prep/build_datasets.py --dataset yelp --skip-convert
        } else {
            uv run scripts/data_prep/build_datasets.py --dataset $dataset
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to build dataset: $dataset"
            exit $LASTEXITCODE
        }
    }

    # 2. Preprocess dataset (creates clean_text and clean_text_stemmed)
    Write-Host "[2/6] Preprocessing dataset (Dual Version: Unstemmed + Stemmed)..." -ForegroundColor Gray
    uv run scripts/data_prep/preprocess_datasets.py --dataset $dataset
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to preprocess dataset: $dataset"
        exit $LASTEXITCODE
    }

    # 3. Generate sentence embeddings for both clean_text and clean_text_stemmed
    Write-Host "[3/6] Generating embeddings for clean_text and clean_text_stemmed..." -ForegroundColor Gray
    uv run scripts/data_prep/generate_embeddings.py --dataset $dataset --columns clean_text clean_text_stemmed
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to generate embeddings: $dataset"
        exit $LASTEXITCODE
    }

    # 4. Tokenize both text columns with the topic-word analyzer for STM
    Write-Host "[4/6] Tokenizing texts for the STM bag-of-words..." -ForegroundColor Gray
    uv run python scripts/data_prep/build_bow_tokens.py --dataset $dataset
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to tokenize $dataset"
        exit $LASTEXITCODE
    }

    # 5. Build Unstemmed R BoW / STM representations
    Write-Host "[5/6] Building R BoW and STM representations (Unstemmed)..." -ForegroundColor Gray
    Rscript scripts/r_scripts/build_bow.R --dataset $dataset --text_col clean_text
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build unstemmed BoW for $dataset"
        exit $LASTEXITCODE
    }

    # 6. Build Stemmed R BoW / STM representations
    Write-Host "[6/6] Building R BoW and STM representations (Stemmed)..." -ForegroundColor Gray
    Rscript scripts/r_scripts/build_bow.R --dataset $dataset --text_col clean_text_stemmed --output_suffix _stemmed
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build stemmed BoW for $dataset"
        exit $LASTEXITCODE
    }
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " All representations generated successfully!" -ForegroundColor Green
Write-Host " Dual versions (Unstemmed and Stemmed) created for all datasets." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
