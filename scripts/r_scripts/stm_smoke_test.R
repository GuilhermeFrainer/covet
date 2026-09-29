# Mock STM Pipeline: Training and Exporting for Python Evaluation
library(stm)
library(arrow)

cat("==========================================\n")
cat("Starting Mock STM Pipeline inside Docker\n")
cat("Library Paths:\n")
print(.libPaths())
cat("==========================================\n")

# Setup paths (relative to /app)
input_file <- "data/processed/trump_stm_data.rds"
output_dir <- "results/test_handoff"
model_dir  <- "models/test_handoff"

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
if (!dir.exists(model_dir)) dir.create(model_dir, recursive = TRUE)

cat("Checking for data at:", input_file, "\n")

if (file.exists(input_file)) {
    cat("Data file found! Loading RDS...\n")
    stm_data <- readRDS(input_file)
    
    # Tiny sample for speed
    n_sample <- min(200, length(stm_data$documents))
    short_docs <- stm_data$documents[1:n_sample]
    short_meta <- stm_data$meta[1:n_sample, ]
    
    cat("Cleaning subset with prepDocuments...\n")
    # prepDocuments ensures indices are 1-indexed and sequential
    processed <- prepDocuments(short_docs, stm_data$vocab, short_meta, lower.thresh = 1)
    
    cat(sprintf("After cleaning: %d documents, %d words in vocab.\n", 
                length(processed$documents), length(processed$vocab)))

    cat("Training 5-topic model (K=5)...\n")
    start_time <- Sys.time()
    model <- stm(
        documents = processed$documents,
        vocab = processed$vocab,
        K = 5,
        data = processed$meta,
        init.type = "Spectral",
        max.em.its = 5,
        verbose = TRUE
    )
    end_time <- Sys.time()
    duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    cat("\nTraining successful! Exporting handoff files for Python...\n")
    
    # 1. Beta (Topic-Word)
    beta <- exp(model$beta$logbeta[[1]])
    write_parquet(as.data.frame(beta), file.path(output_dir, "beta.parquet"))
    cat(" - Exported beta.parquet\n")
    
    # 2. Theta (Document-Topic)
    theta <- as.data.frame(model$theta)
    write_parquet(theta, file.path(output_dir, "theta.parquet"))
    cat(" - Exported theta.parquet\n")
    
    # 3. Vocab
    writeLines(model$vocab, file.path(output_dir, "vocab.txt"))
    cat(" - Exported vocab.txt\n")
    
    # 4. Duration
    writeLines(as.character(duration), file.path(output_dir, "duration.txt"))
    cat(" - Exported duration.txt\n")
    
    # 5. Full Model RDS
    saveRDS(model, file.path(model_dir, "mock_model.rds"))
    cat(" - Exported mock_model.rds\n")
    
    cat("==========================================\n")
    cat("Handoff files are ready in results/test_handoff/\n")
    cat("==========================================\n")
} else {
    cat("ERROR: Data file NOT found at:", input_file, "\n")
    quit(status = 1)
}
