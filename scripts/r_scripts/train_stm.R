# Trains one STM and exports what scripts/experiments/run_stm.py evaluates.
#
# Runs inside the lightweight STM image (Dockerfile.stm), so it uses only stm,
# its dependency data.table, and optparse. Outputs in --output_dir:
#   beta.csv.gz     K x V topic-word probabilities
#   theta.csv.gz    D x K document-topic proportions
#   vocab.txt       one word per line, in beta's column order
#   doc_index.txt   the `index` of each trained document, in theta's row order
#   duration.txt    training time in seconds
library(stm)
library(data.table)
library(optparse)

main <- function() {
    option_list <- list(
        make_option(c("-r", "--rds_path"), type = "character", help = "Path to stm_data.rds"),
        make_option(c("-k", "--k"), type = "integer", help = "Number of topics"),
        make_option(c("-i", "--indices_path"), type = "character", default = NULL,
                    help = "Text file with one document `index` per line to train on"),
        make_option(c("-o", "--output_dir"), type = "character", help = "Directory to save outputs"),
        make_option(c("-s", "--seed"), type = "integer", default = 42, help = "Random seed"),
        make_option(c("--model_path"), type = "character", help = "Path to save the full model RDS"),
        make_option(c("--prevalence_formula"), type = "character", default = NULL,
                    help = "Prevalence formula (e.g., '~ as.factor(party)')")
    )

    opt <- parse_args(OptionParser(option_list = option_list))

    cat("\n=== [STM TRAINING PRE-FLIGHT CHECK] ===\n")
    cat("Working Directory: ", getwd(), "\n")
    for (name in names(opt)) {
        cat(sprintf("  --%-18s: %s\n", name, as.character(opt[[name]])))
    }
    if (is.null(opt$rds_path) || !file.exists(opt$rds_path)) {
        stop(sprintf("rds_path not found: '%s'. Check your volume mounts.", opt$rds_path))
    }
    if (is.null(opt$output_dir) || is.null(opt$model_path)) {
        stop("output_dir and model_path are required.")
    }
    dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(dirname(opt$model_path), recursive = TRUE, showWarnings = FALSE)
    cat("======================================\n\n")

    set.seed(opt$seed)

    stm_data <- readRDS(opt$rds_path)
    cat(sprintf("Loaded RDS data with %d documents.\n", length(stm_data$documents)))

    if (!is.null(opt$indices_path)) {
        cat("Applying sampling...\n")
        indices <- as.numeric(readLines(opt$indices_path))
        keep_mask <- stm_data$meta$index %in% indices
        cat(sprintf("Keeping %d out of %d documents after sampling.\n", sum(keep_mask), length(keep_mask)))
        # Re-index the vocabulary so words absent from the sample are dropped
        prepped <- prepDocuments(stm_data$documents[keep_mask], stm_data$vocab,
                                 stm_data$meta[keep_mask, ], lower.thresh = 0, verbose = FALSE)
        stm_data$documents <- prepped$documents
        stm_data$vocab <- prepped$vocab
        stm_data$meta <- prepped$meta
    }

    cat(sprintf("Documents: %d\n", length(stm_data$documents)))
    cat(sprintf("Vocab size: %d\n", length(stm_data$vocab)))
    cat(sprintf("Metadata columns: %s\n", paste(names(stm_data$meta), collapse = ", ")))

    formula <- NULL
    if (!is.null(opt$prevalence_formula)) {
        cat(sprintf("Prevalence formula: %s\n", opt$prevalence_formula))
        formula <- as.formula(opt$prevalence_formula)
    } else {
        cat("No prevalence formula provided. Using vanilla STM.\n")
    }

    if (length(stm_data$documents) == 0) {
        stop("No documents left to train the model.")
    }

    cat(sprintf("Training STM model with K=%d...\n", opt$k))
    start_time <- Sys.time()
    model <- stm(
        documents = stm_data$documents,
        vocab = stm_data$vocab,
        K = opt$k,
        prevalence = formula,
        data = stm_data$meta,
        init.type = "Spectral",
        seed = opt$seed,
        verbose = TRUE
    )
    duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    cat(sprintf("Training finished in %.2f seconds.\n", duration))
    cat(sprintf("Generated topics: %d\n", model$settings$dim$K))
    if (!is.null(model$settings$covariates$formula)) {
        cat(sprintf("Formula stored in model: %s\n",
                    Reduce(paste, deparse(model$settings$covariates$formula))))
    }

    saveRDS(model, file = opt$model_path)

    # Without content covariates, logbeta has a single K x V matrix
    fwrite(as.data.table(exp(model$beta$logbeta[[1]])),
           file.path(opt$output_dir, "beta.csv.gz"))
    fwrite(as.data.table(model$theta), file.path(opt$output_dir, "theta.csv.gz"))
    writeLines(model$vocab, file.path(opt$output_dir, "vocab.txt"), useBytes = TRUE)
    writeLines(format(stm_data$meta$index, scientific = FALSE, trim = TRUE),
               file.path(opt$output_dir, "doc_index.txt"))
    writeLines(as.character(duration), file.path(opt$output_dir, "duration.txt"))

    cat("Done!\n")
}

if (sys.nframe() == 0) {
    main()
}
