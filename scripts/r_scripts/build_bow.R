# Builds the STM bag-of-words from tokens produced in Python.
#
# Tokenization happens in scripts/data_prep/build_bow_tokens.py with the
# analyzer BERTopic variants use for topic words, so STM and the neural models
# share one vocabulary. This script only splits those tokens on whitespace;
# it never lowercases, filters, or stems.
library(arrow)
library(quanteda)
library(optparse)
library(here)

main <- function() {
    option_list <- list(
        make_option(
            c("-d", "--dataset"),
            type = "character",
            help = "Dataset name (e.g., 'fed', 'anes', 'gadarian', 'trump_s25000', 'yelp_s10000')"),
        make_option(
            c("-i", "--input"),
            type = "character",
            help = "Input parquet path (default: data/interim/<dataset>_bow_tokens.parquet)"),
        make_option(
            c("-t", "--text_col"),
            type = "character",
            default = "clean_text",
            help = "Text column whose tokens to use (default: 'clean_text')"),
        make_option(
            c("--output_suffix"),
            type = "character",
            default = "",
            help = "Suffix for output files (default: '_stemmed' for clean_text_stemmed)")
    )

    opt <- parse_args(OptionParser(option_list = option_list))

    if (is.null(opt$dataset)) {
        stop("Dataset name is required. Use --dataset <name>")
    }

    if (opt$text_col == "clean_text_stemmed" && opt$output_suffix == "") {
        opt$output_suffix <- "_stemmed"
    }

    input_path <- opt$input
    if (is.null(input_path)) {
        input_path <- here::here("data", "interim", paste0(opt$dataset, "_bow_tokens.parquet"))
    }
    if (!file.exists(input_path)) {
        stop(paste("Input file not found:", input_path,
                   "- run scripts/data_prep/build_bow_tokens.py first."))
    }

    token_col <- paste0("bow_tokens_", opt$text_col)

    cat(sprintf("Processing dataset: %s\n", opt$dataset))
    cat(sprintf("Input path: %s\n", input_path))
    cat(sprintf("Token column: %s\n", token_col))

    # Avoid issues with dictionary/factor conversion
    options(arrow.use_factors = FALSE)

    tab <- arrow::read_parquet(input_path, as_data_frame = FALSE)
    if (!token_col %in% names(tab)) {
        stop(sprintf("Column '%s' not found in %s.", token_col, input_path))
    }

    # Cast dictionary columns to string (fixes conversion issues)
    schema <- tab$schema
    dict_cols <- names(tab)[sapply(names(tab), function(x) inherits(schema[[x]]$type, "DictionaryType"))]
    for (col in dict_cols) {
        tab[[col]] <- tab[[col]]$cast(arrow::utf8())
    }

    data <- as.data.frame(tab)
    cat(sprintf("Loaded %d rows.\n", nrow(data)))

    # Documents whose tokens were all stopwords cannot enter a bag-of-words
    empty <- !nzchar(data[[token_col]])
    cat(sprintf("Dropping %d documents with no tokens.\n", sum(empty)))
    data <- data[!empty, ]

    data$bow_text <- data[[token_col]]
    data <- data[, !startsWith(names(data), "bow_tokens_")]

    corp <- quanteda::corpus(data, text_field = "bow_text")
    toks <- quanteda::tokens(corp, what = "fastestword")
    dfm_obj <- quanteda::dfm(toks, tolower = FALSE)

    stm_data <- quanteda::convert(dfm_obj, to = "stm")
    if (length(stm_data$documents) != nrow(data)) {
        stop(sprintf("STM conversion dropped documents: %d of %d kept.",
                     length(stm_data$documents), nrow(data)))
    }
    cat(sprintf("Documents: %d, vocabulary: %d\n", nrow(data), length(stm_data$vocab)))

    prefix <- paste0(opt$dataset, opt$output_suffix)

    rds_output_path <- here::here("data", "processed", paste0(prefix, "_stm_data.rds"))
    saveRDS(stm_data, file = rds_output_path)
    cat(sprintf("Saved RDS to: %s\n", rds_output_path))

    parquet_output_path <- here::here("data", "processed", paste0(prefix, "_bow.parquet"))
    arrow::write_parquet(data, parquet_output_path)
    cat(sprintf("Saved Parquet to: %s\n", parquet_output_path))

    cat("Done!\n")
}

if (sys.nframe() == 0) {
    main()
}
