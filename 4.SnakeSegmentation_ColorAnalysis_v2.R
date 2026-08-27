# ================================================================
# Snake color extraction pipeline
# CIELAB illumination standardization + k-means clustering (k = 2)
# ================================================================

# --- 1. Libraries --- ####
library(imager)
library(dplyr)

# --- 2. File system set up --- ####
img_dir <- "C:/Users/pdeschepper/OneDrive - Institute of Natural Sciences/Desktop/PERSONAL/DeepLearning/vipera_spatialcolors/Gbif_sourceimages/Extracted_snakes_pytorch"

images <- list.files(img_dir, pattern = "\\.png$", ignore.case = TRUE, full.names = TRUE)
stopifnot(length(images) > 0)

out_dir <- file.path(img_dir, "ColorExtraction_results")
if (!dir.exists(out_dir)) dir.create(out_dir)

# Optional QC step: write out the standardized PNGs so you can visually
# inspect the correction. Off by default -- not needed for the CSV output.
SAVE_STANDARDIZED_IMAGES <- FALSE
if (SAVE_STANDARDIZED_IMAGES) {
  std_dir <- file.path(img_dir, "Standardized")
  if (!dir.exists(std_dir)) dir.create(std_dir)
}

# --- 3. Standardization function --- ####
TARGET_L_MEAN   <- 65     # reference brightness level all images are shifted to
TARGET_L_SD     <- 15     # only used if STANDARDIZE_SD <- TRUE
STANDARDIZE_SD  <- FALSE  # see notes: mean-only correction preserves real
# pattern contrast; SD-matching would flatten it

# Standardizes the L* (lightness) channel of a single image to correct for
# differing exposure/illumination across GBIF photos, while leaving a*/b*
# (the actual hue/chroma information) untouched. Operates only on pixels
# where alpha > 0 (i.e. actual snake pixels, since the bounding-box crop
# still contains transparent pixels outside the silhouette).
standardize_L_channel <- function(img_path, target_mean, target_sd = NULL,
                                  standardize_sd = FALSE) {
  
  img_full <- imager::load.image(img_path)
  has_alpha <- dim(img_full)[4] == 4
  
  if (has_alpha) {
    img_rgb <- imager::channel(img_full, 1:3)
    alpha_channel <- imager::channel(img_full, 4)
    mask <- as.vector(alpha_channel) > 0.5
  } else {
    img_rgb <- img_full
    alpha_channel <- NULL
    mask <- rep(TRUE, prod(dim(img_full)[1:2]))
  }
  
  if (sum(mask) < 50) {
    stop("Fewer than 50 non-transparent (snake) pixels found -- check the crop.")
  }
  
  img_lab <- imager::RGBtoLab(img_rgb)
  L_channel <- imager::channel(img_lab, 1)
  a_channel <- imager::channel(img_lab, 2)
  b_channel <- imager::channel(img_lab, 3)
  
  L_vals <- as.vector(L_channel)[mask]
  L_vals <- L_vals[is.finite(L_vals)]
  
  # Robust center/spread: median + MAD are far less influenced by a small
  # number of specular highlight or deep-shadow pixels at the scale edges
  # than a plain mean/sd would be.
  current_center <- stats::median(L_vals)
  current_spread <- stats::mad(L_vals, constant = 1.4826)
  
  if (!is.finite(current_spread) || current_spread == 0) {
    warning(paste("MAD zero/NA for:", basename(img_path), "- shifting to target mean only."))
    standardized_L <- L_channel - current_center + target_mean
  } else if (standardize_sd) {
    standardized_L <- ((L_channel - current_center) / current_spread) * target_sd + target_mean
  } else {
    # Mean-only correction: shift brightness to a common reference level,
    # but leave each image's native contrast untouched. This corrects
    # exposure/illumination differences without erasing real pattern
    # contrast on the snake (blotching, banding, etc.).
    standardized_L <- L_channel - current_center + target_mean
  }
  
  standardized_L[standardized_L < 0] <- 0
  standardized_L[standardized_L > 100] <- 100
  
  standardized_img_lab <- imager::imappend(list(standardized_L, a_channel, b_channel), "c")
  standardized_img_rgb <- imager::LabtoRGB(standardized_img_lab)
  
  # Clamp to valid sRGB gamut -- Lab -> RGB round trips can overshoot [0,1]
  standardized_img_rgb[standardized_img_rgb < 0] <- 0
  standardized_img_rgb[standardized_img_rgb > 1] <- 1
  
  list(
    rgb = standardized_img_rgb,
    lab = standardized_img_lab,
    alpha = alpha_channel,
    mask = mask,
    has_alpha = has_alpha
  )
}

# --- 4. Clustering function (k = 2, on standardized Lab pixels) --- ####
SAMPLE_SIZE    <- 10000
N_CLUSTERS     <- 2
KMEANS_NSTART  <- 10   # multiple random starts to avoid a bad local optimum

# Runs k-means directly on the standardized Lab pixel values (masked to
# snake-only pixels, subsampled for speed), and returns cluster centers as
# Lab coordinates plus sRGB hex, ordered largest-proportion first.
extract_lab_clusters <- function(std_result, n = N_CLUSTERS,
                                 sample.size = SAMPLE_SIZE, nstart = KMEANS_NSTART) {
  
  lab_img <- std_result$lab
  mask <- std_result$mask
  
  L <- as.vector(imager::channel(lab_img, 1))[mask]
  a <- as.vector(imager::channel(lab_img, 2))[mask]
  b <- as.vector(imager::channel(lab_img, 3))[mask]
  
  lab_matrix <- cbind(L, a, b)
  lab_matrix <- lab_matrix[stats::complete.cases(lab_matrix), , drop = FALSE]
  
  if (nrow(lab_matrix) < n) {
    stop("Not enough valid pixels to cluster.")
  }
  
  if (nrow(lab_matrix) > sample.size) {
    lab_matrix <- lab_matrix[sample.int(nrow(lab_matrix), sample.size), ]
  }
  
  km <- stats::kmeans(lab_matrix, centers = n, nstart = nstart, iter.max = 50)
  
  sizes <- km$size / sum(km$size)
  ord <- order(sizes, decreasing = TRUE)  # largest cluster first, for consistency across images
  
  centers <- km$centers[ord, , drop = FALSE]
  sizes <- sizes[ord]
  
  rgb_vals <- grDevices::convertColor(centers, from = "Lab", to = "sRGB", clip = TRUE)
  rgb_vals[rgb_vals < 0] <- 0
  rgb_vals[rgb_vals > 1] <- 1
  hex <- grDevices::rgb(rgb_vals[, 1], rgb_vals[, 2], rgb_vals[, 3])
  
  data.frame(
    cluster    = seq_len(n),
    proportion = sizes,
    L = centers[, 1],
    a = centers[, 2],
    b = centers[, 3],
    hex = hex,
    stringsAsFactors = FALSE
  )
}

# --- 5. Run pipeline over all images and write CSV --- ####
results <- vector("list", length(images))
failed <- character(0)

for (i in seq_along(images)) {
  img_path <- images[i]
  photo_id <- tools::file_path_sans_ext(basename(img_path))
  cat(sprintf("[%d/%d] Processing %s\n", i, length(images), basename(img_path)))
  
  res <- tryCatch({
    std <- standardize_L_channel(img_path, target_mean = TARGET_L_MEAN,
                                 target_sd = TARGET_L_SD, standardize_sd = STANDARDIZE_SD)
    
    if (SAVE_STANDARDIZED_IMAGES) {
      out_img <- if (std$has_alpha) imager::imappend(list(std$rgb, std$alpha), "c") else std$rgb
      imager::save.image(out_img, file.path(std_dir, basename(img_path)))
    }
    
    clust_df <- extract_lab_clusters(std)
    clust_df$photo_id <- photo_id
    clust_df$file <- basename(img_path)
    clust_df
    
  }, error = function(e) {
    warning(paste("Failed on", basename(img_path), ":", e$message))
    failed <<- c(failed, basename(img_path))
    NULL
  })
  
  results[[i]] <- res
}

results_df <- dplyr::bind_rows(results) |>
  dplyr::select(photo_id, file, cluster, proportion, L, a, b, hex)

write.csv(results_df, file.path(out_dir, "snake_color_clusters.csv"), row.names = FALSE)

if (length(failed) > 0) {
  cat("\nThe following images failed and were skipped:\n")
  print(failed)
  writeLines(failed, file.path(out_dir, "failed_images.txt"))
}

cat(sprintf(
  "\nDone. Processed %d/%d images successfully. Results written to:\n%s\n",
  nrow(results_df) / N_CLUSTERS, length(images),
  file.path(out_dir, "snake_color_clusters.csv")
))


# --- 6. Exploration of output --- ####
# Code for visual comparison after standardization

compare_standardization <- function(img_path) {
  original <- imager::load.image(img_path)
  std <- standardize_L_channel(img_path, target_mean = TARGET_L_MEAN,
                               target_sd = TARGET_L_SD, standardize_sd = STANDARDIZE_SD)
  
  standardized <- if (std$has_alpha) {
    imager::imappend(list(std$rgb, std$alpha), "c")
  } else {
    std$rgb
  }
  
  old_par <- par(mfrow = c(1, 2), mar = c(1, 1, 2, 1))
  plot(original, axes = FALSE, main = "Original")
  plot(standardized, axes = FALSE, main = "Standardized")
  par(old_par)
}

# Usage:
compare_standardization(images[8])
