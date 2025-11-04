library(colordistance)
library(ape)
library(magick)
library(pheatmap)
library(imager) 
library(stringr)
library(dplyr)

# --- Set working dir where images are --- ####
img_dir <- "C:/Users/pdeschepper/Desktop/PERSONAL/DeepLearning/ImageSegmentation/Snakes_ImageSegmentation_keras/Vipera_segmentation_test_dataset/Extracted_snakes/"
images <- list.files(img_dir, pattern = "*.png$", full.names = TRUE)
# Creat directory for standardized images
std <- paste0(img_dir, "/Standardized")
if (dir.exists(std)) {} else {dir.create(std)}



# --- Standardization function --- ####
TARGET_L_MEAN <- 50
TARGET_L_SD <- 15

standardize_L_channel <- function(img_path, target_mean, target_sd) {
  result <- tryCatch({
    # 1. Read image using imager (allow 4 channels to capture Alpha)
    img_rgb_full <- imager::load.image(img_path)
    
    # Check for alpha channel
    has_alpha <- dim(img_rgb_full)[4] == 4
    
    # 2. Separate color channels (RGB) from Alpha
    if (has_alpha) {
      img_rgb <- imager::channel(img_rgb_full, 1:3) 
      alpha_channel <- imager::channel(img_rgb_full, 4)
    } else {
      img_rgb <- img_rgb_full
      alpha_channel <- NULL # No alpha channel in the original
    }
    
    # 3. Convert to CIELAB (Only on the 3 color channels)
    img_lab <- RGBtoLab(img_rgb)
    
    # 4. Extract L*, a*, b* channels
    L_channel <- channel(img_lab, 1) 
    a_channel <- channel(img_lab, 2)
    b_channel <- channel(img_lab, 3)
    
    # 5. Calculate current statistics of the L* channel (excluding non-finite values)
    L_vals <- as.vector(L_channel)
    L_vals <- L_vals[is.finite(L_vals)]
    
    current_mean <- mean(L_vals, na.rm = TRUE)
    current_sd <- sd(L_vals, na.rm = TRUE)
    
    # 6. Standardize L* Channel
    if (current_sd == 0 || is.na(current_sd)) {
      warning(paste("SD zero/NA for:", basename(img_path), "- setting L* to target mean."))
      standardized_L <- target_mean
    } else {
      standardized_L <- ((L_channel - current_mean) / current_sd) * target_sd + target_mean
    }
    
    # 7. Constrain L* values to [0, 100] using base R
    standardized_L[standardized_L < 0] <- 0
    standardized_L[standardized_L > 100] <- 100
    
    # 8. Recombine standardized L* with original a* and b*
    standardized_img_lab <- imager::imappend(list(standardized_L, a_channel, b_channel), "c")
    standardized_img_rgb <- imager::LabtoRGB(standardized_img_lab)
    
    # 9. Re-append the original Alpha Channel if it existed
    if (has_alpha) {
      # Append the alpha channel as the 4th channel
      final_img_rgba <- imager::imappend(list(standardized_img_rgb, alpha_channel), "c")
    } else {
      final_img_rgba <- standardized_img_rgb
    }
    
    # 10. Save the final image
    standardized_file <- img_path %>% str_replace("(Extracted_snakes/)", "\\1Standardized/")
    save.image(final_img_rgba, file = file.path(standardized_file))
    
    return(standardized_file)
    
  }, error = function(e) {
    warning(paste("CRITICAL ERROR processing", basename(img_path), ":", e$message, "- Using original file path as fallback."))
    return(img_path)
  })
  
  return(result)
}

# --- Run the standardization step with the revised function --- ####
cat("Standardizing L* channel across images with error handling...\n")
standardized_images <- lapply(images, standardize_L_channel, 
                              target_mean = TARGET_L_MEAN, 
                              target_sd = TARGET_L_SD)
cat("Standardization pass complete. Check console for warnings about skipped images.\n")

# Code for visual comparison after standardization
compare_visual_base_r_simple <- function(original_path, standardized_path) {
  
  # 1. Read the PNG files directly into an array using the 'png' package
  img_orig <- readPNG(original_path)
  img_std <- readPNG(standardized_path)
  
  # 2. Set up the plotting area for 1 row and 2 columns
  # Set the margins low (mai) and adjust the image size (mar)
  par(mfrow = c(1, 2), mar = c(1, 1, 1, 1))
  
  # 3. Plot the original image
  plot.new()
  # Set the plot range to match the image aspect ratio
  plot.window(xlim = c(0, 1), ylim = c(0, dim(img_orig)[1] / dim(img_orig)[2]))
  rasterImage(img_orig, 0, 0, 1, dim(img_orig)[1] / dim(img_orig)[2])
  title("Original")
  
  # 4. Plot the standardized image
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, dim(img_std)[1] / dim(img_std)[2]))
  rasterImage(img_std, 0, 0, 1, dim(img_std)[1] / dim(img_std)[2])
  title("L* Standardized")
  
  # 5. Reset plotting parameters after plotting is complete
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1) 
}

compare_visual_base_r_simple(images[1], standardized_images[[1]])

# ---- Kmeans color clustering for a single image ---- ####
# Use the first standardized image path
clust <- colordistance::getKMeanColors(standardized_images[[9]], n = 3, sample.size = 10000,
                                       color.space = "lab", ref.white = "D65")

# ---- Analyze all images to plot distance matrix ---- ####
# Use the standardized image paths
lab_hist_list <- getLabHistList(standardized_images, bins = 2, sample.size = 10000,
                                ref.white = "D65", lower = rep(0.8, 3), upper = rep(1, 3),
                                plotting = FALSE, pausing = FALSE)

# Compute and plot color distance matrix
lab_dist_matrix <- getColorDistanceMatrix(lab_hist_list, method = "emd", plotting = FALSE)
# Reset row/column names to original image names for clarity in the plot
rownames(lab_dist_matrix) <- colnames(lab_dist_matrix) <- names(standardized_images)
pheatmap(lab_dist_matrix)

# ---- Plot dendrogram with barplots at tips ----

# Convert to dist object
lab_dist <- as.dist(lab_dist_matrix)

# Hierarchical clustering
hc <- hclust(lab_dist, method = "average")
tree <- as.phylo(hc)

# Extract color clusters for each image (using standardized images)
clust_list <- lapply(standardized_images, function(img_path) {
  # The input is the path to the standardized temp file
  cl <- colordistance::getKMeanColors(
    img_path, n = 2, sample.size = 10000,
    color.space = "lab", ref.white = "D65"
  )
  sizes <- cl$size / sum(cl$size)  # normalize sizes
  cols <- apply(cl$centers, 1, function(x) {
    # Convert LAB to sRGB hex for plotting
    rgb_vals <- grDevices::convertColor(matrix(x, nrow=1), from="Lab", to="sRGB")
    rgb(rgb_vals[1], rgb_vals[2], rgb_vals[3], maxColorValue = 1)
  })
  list(sizes = sizes, cols = cols)
})
names(clust_list) <- names(standardized_images) # Use original image names

# Plot the dendrogram (rest of the plotting code is unchanged)
plot(tree, show.tip.label = FALSE, cex = 0.7,
     main = "Dendrogram of image color dissimilarity (L* Normalized)")

# Extract coordinates of tips
coords <- get("last_plot.phylo", envir = .PlotPhyloEnv)
tip_x <- coords$xx[1:length(tree$tip.label)]
tip_y <- coords$yy[1:length(tree$tip.label)]

# Determine bar size relative to plot
x_range <- diff(range(coords$xx))
y_range <- diff(range(coords$yy))
bar_length <- 0.07 * x_range
bar_height <- 0.02 * y_range

# Draw color bars
for (i in seq_along(tree$tip.label)) {
  img_name <- tree$tip.label[i]
  cb <- clust_list[[img_name]]
  
  if (is.null(cb)) next
  
  x_start <- tip_x[i] - bar_length / 2
  x_end <- x_start
  
  for (j in seq_along(cb$sizes)) {
    width <- cb$sizes[j] * bar_length
    rect(x_end, tip_y[i] - bar_height/2,
         x_end + width, tip_y[i] + bar_height/2,
         col = cb$cols[j], border = NA)
    x_end <- x_end + width
  }
}


# ---- Plot dendrogram with images at tips ----

# Create thumbnails (Use original images for display, standardized images for clustering)
# This is a key design choice: cluster on standardized color, but show original images.
thumbnails <- lapply(images, function(path) {
  image_read(path) |> image_scale("120x120")
})
names(thumbnails) <- basename(images)

# Plot tree (tree is based on normalized lab_dist_matrix)
plot(tree, show.tip.label = FALSE, main = "Color similarity dendrogram (L* Normalized)")

# Get coordinates
coords <- get("last_plot.phylo", envir = .PlotPhyloEnv)
tip_x <- coords$xx[1:length(tree$tip.label)]
tip_y <- coords$yy[1:length(tree$tip.label)]

# Size relative to tree
x_range <- diff(range(coords$xx))
y_range <- diff(range(coords$yy))
x_size <- 0.04 * x_range
y_size <- 0.04 * y_range

# Match thumbnails to tip order
thumbnails_ordered <- thumbnails[match(tree$tip.label, hc$labels)]

# Add thumbnails
for (i in seq_along(tree$tip.label)) {
  img_raster <- as.raster(thumbnails_ordered[[i]])
  rasterImage(
    img_raster,
    tip_x[i] - x_size, tip_y[i] - y_size,
    tip_x[i] + x_size, tip_y[i] + y_size
  )
}

# --- Cleanup ---
# Remove the temporary standardized image files
file.remove(unlist(standardized_images))