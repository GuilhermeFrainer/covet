library(arrow)
library(dplyr)
library(quanteda)
library(SnowballC)
library(optparse)
library(here)

main <- function() {
    option_list <- list(
        make_option(
            c("-d", "--dataset"),
            type = "character",
            help = "Dataset name (e.g., 'fed', 'anes', 'gadarian', 'trump', 'yelp')"),
        make_option(
            c("-i", "--input"),
            type = "character",
            help = "Input parquet file path (optional)"),
        make_option(
            c("-t", "--text_col"),
            type = "character",
            default = "text",
            help = "Column containing the text (default: 'text')"),
        make_option(
            c("--min_freq"),
            type = "integer",
            default = 1,
            help = "Minimum document frequency for terms (default: 1)"),
        make_option(
            c("--max_freq_pct"),
            type = "double",
            default = 1.0,
            help = "Maximum document frequency percentage for terms (default: 1.0)"),
        make_option(
            c("--output_suffix"),
            type = "character",
            default = "",
            help = "Suffix for output files (e.g., '_stemmed')"),
        make_option(
            c("--no_stem"),
            action = "store_true",
            default = FALSE,
            help = "Deprecated: Stemming is now handled in Python preprocessing"),
        make_option(
            c("--deduplicate"),
            action = "store_true",
            default = FALSE,
            help = "Remove duplicate documents (default: FALSE)"),
        make_option(
            c("--sample"),
            type = "integer",
            help = "Sample size (default: NULL, no sampling)"),
        make_option(
            c("--seed"),
            type = "integer",
            default = 36201624,
            help = "Random seed for sampling (default: 36201624)")
    )
    
    opt <- parse_args(OptionParser(option_list = option_list))
    
    if (is.null(opt$dataset)) {
        stop("Dataset name is required. Use --dataset <name>")
    }
    
    # Auto set output suffix if text_col is clean_text_stemmed and suffix not explicitly set
    if (opt$text_col == "clean_text_stemmed" && opt$output_suffix == "") {
        opt$output_suffix <- "_stemmed"
    }
    
    # Determine input path
    if (is.null(opt$input)) {
        # Default mapping for interim files
        file_mapping <- list(
            fed = "fed_processed.parquet",
            anes = "anes_processed.parquet",
            gadarian = "gadarian_processed.parquet",
            trump = "trump_processed.parquet",
            yelp = "yelp_processed.parquet",
            yelp_s10000 = "yelp_s10000_processed.parquet"
        )
        
        file_name <- file_mapping[[opt$dataset]]
        if (is.null(file_name)) {
            # Sampled datasets (e.g. trump_s25000) follow the same naming convention
            file_name <- paste0(opt$dataset, "_processed.parquet")
        }
        input_path <- here::here("data", "interim", file_name)
    } else {
        input_path <- opt$input
    }
    
    if (!file.exists(input_path)) {
        stop(paste("Input file not found:", input_path))
    }
    
    cat(sprintf("Processing dataset: %s\n", opt$dataset))
    cat(sprintf("Input path: %s\n", input_path))
    cat(sprintf("Text column: %s\n", opt$text_col))
    
    # Set arrow option to avoid issues with dictionary/factor conversion
    options(arrow.use_factors = FALSE)
    
    # Selective loading: exclude embeddings if they exist in input
    cat("Loading data...\n")
    tab <- arrow::read_parquet(input_path, as_data_frame = FALSE)
    all_cols <- names(tab)
    meta_cols <- all_cols[!grepl("embedding", all_cols)]
    
    # Select columns on the table
    tab <- tab[, meta_cols]
    
    # Identify dictionary columns to cast them to string (fixes conversion issues)
    schema <- tab$schema
    dict_cols <- meta_cols[sapply(meta_cols, function(x) inherits(schema[[x]]$type, "DictionaryType"))]
    
    if (length(dict_cols) > 0) {
        cat(sprintf("Casting dictionary columns to string: %s\n", paste(dict_cols, collapse=", ")))
        for (col in dict_cols) {
            # Use cast on the ChunkedArray
            tab[[col]] <- tab[[col]]$cast(arrow::utf8())
        }
    }
    
    data <- as.data.frame(tab)
    
    cat(sprintf("Loaded %d rows.\n", nrow(data)))

    # Deduplication
    if (opt$deduplicate) {
        cat("Deduplicating based on text column...\n")
        initial_count <- nrow(data)
        
        if ("date" %in% names(data)) {
            # Sort by date and keep first occurrence
            data <- data %>%
                arrange(date) %>%
                distinct(!!rlang::sym(opt$text_col), .keep_all = TRUE)
        } else {
            data <- data %>%
                distinct(!!rlang::sym(opt$text_col), .keep_all = TRUE)
        }
        
        final_count <- nrow(data)
        cat(sprintf("Deduplication complete. Dropped %d duplicate rows.\n", initial_count - final_count))
    }

    # Sampling
    if (!is.null(opt$sample)) {
        if (opt$sample < nrow(data)) {
            cat(sprintf("Sampling %d rows (seed=%d)...\n", opt$sample, opt$seed))
            set.seed(opt$seed)
            data <- data[sample(nrow(data), opt$sample), ]
        } else {
            cat(sprintf("Sample size %d >= dataset size %d. No sampling applied.\n", opt$sample, nrow(data)))
            opt$sample <- NULL # Reset to avoid suffixing
        }
    }
    
    # Tokenization and Cleaning
    cat("Tokenizing text from Python preprocessed text column...\n")
    if (!opt$text_col %in% names(data)) {
        stop(sprintf("Column '%s' not found in dataset.", opt$text_col))
    }
    
    corp <- quanteda::corpus(data, text_field = opt$text_col)
    
    if (opt$text_col == "clean_text_stemmed") {
        # Python has already lowercased, stripped punctuation, removed stopwords, and stemmed
        # Tokenize by splitting on whitespace without applying R-side text modifications
        toks <- quanteda::tokens(corp,
                                 split_hyphens = FALSE,
                                 remove_punct = FALSE,
                                 remove_symbols = FALSE,
                                 remove_numbers = FALSE,
                                 remove_url = FALSE,
                                 remove_separators = TRUE)
    } else {
        # Unstemmed text (Version 1): lowercase and strip punctuation/symbols for BoW matrix,
        # but do NOT perform stopword removal or stemming in R.
        toks <- quanteda::tokens(corp,
                                 remove_punct = TRUE,
                                 remove_symbols = TRUE,
                                 remove_numbers = TRUE,
                                 remove_url = TRUE,
                                 remove_separators = TRUE)
        toks <- quanteda::tokens_tolower(toks)
    }
    
    # Create DFM
    cat("Creating Document-Feature Matrix...\n")
    dfm_obj <- quanteda::dfm(toks)
    
    # Filtering
    if (opt$min_freq > 1 || opt$max_freq_pct < 1.0) {
        cat(sprintf("Filtering: min_freq=%d, max_freq_pct=%.2f\n", opt$min_freq, opt$max_freq_pct))
        dfm_obj <- quanteda::dfm_trim(dfm_obj,
                                      min_docfreq = opt$min_freq,
                                      max_docfreq = opt$max_freq_pct,
                                      docfreq_type = "prop")
    }

    # Suffix for filenames
    sample_suffix <- if (!is.null(opt$sample)) paste0("_s", opt$sample) else ""
    full_suffix <- paste0(sample_suffix, opt$output_suffix)
    
    # 1. Output RDS for STM
    cat("Preparing and saving STM data (RDS)...\n")
    stm_data <- quanteda::convert(dfm_obj, to = "stm")
    
    rds_output_path <- here::here("data", "processed", paste0(opt$dataset, full_suffix, "_stm_data.rds"))
    saveRDS(stm_data, file = rds_output_path)
    cat(sprintf("Saved RDS to: %s\n", rds_output_path))
    
    # 2. Output BoW Parquet
    cat("Preparing and saving BoW Parquet...\n")
    kept_features <- quanteda::featnames(dfm_obj)
    
    # Filter tokens to keep only those in DFM
    toks_filtered <- quanteda::tokens_select(toks, pattern = kept_features, selection = "keep")
    
    # Collapse tokens back to string
    bow_text_vec <- sapply(toks_filtered, paste, collapse = " ")
    
    if (length(bow_text_vec) != nrow(data)) {
        stop(sprintf("Length mismatch: bow_text (%d) vs data (%d)", length(bow_text_vec), nrow(data)))
    }
    
    data$bow_text <- as.character(bow_text_vec)
    
    parquet_output_path <- here::here("data", "processed", paste0(opt$dataset, full_suffix, "_bow.parquet"))
    arrow::write_parquet(data, parquet_output_path)
    cat(sprintf("Saved Parquet to: %s\n", parquet_output_path))
    
    cat("Done!\n")
}

if (sys.nframe() == 0) {
    main()
}
