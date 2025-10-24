library(colordistance)
library(ape)
library(magick)
library(pheatmap)
library(ape)


# Set the path to your extracted images
img_dir <- "C:/Users/pdeschepper/Desktop/PERSONAL/DeepLearning/ImageSegmentation/Snakes_ImageSegmentation_keras/Vipera_segmentation_test_dataset/Extracted_snakes/"
# Get a list of all the PNG files
images <- list.files(img_dir, pattern = "*.png$", full.names = TRUE)

# ---- Kmeans color clustering for a single image ---- ####
clust <- colordistance::getKMeanColors(images[1], n = 2, sample.size = 10000,
                              color.space = "lab", ref.white = "D65")

# ---- Analyze all images to plot distance matrix ---- ####
lab_hist_list <- getLabHistList(images, bins = 2, sample.size = 10000,
                                ref.white = "D65", lower = rep(0.8, 3), upper = rep(1, 3),
                                plotting = FALSE, pausing = FALSE)

# Compute and plot color distance matrix
lab_dist_matrix <- getColorDistanceMatrix(lab_hist_list, method = "emd", plotting = FALSE)
pheatmap(lab_dist_matrix)

# ---- Plot dendrogram with barplots at tips ---- ####

# Convert to dist object
lab_dist <- as.dist(lab_dist_matrix)

# Hierarchical clustering
hc <- hclust(lab_dist, method = "average")
tree <- as.phylo(hc)

# Extract color clusters for each image 
clust_list <- lapply(images, function(img) {
  cl <- colordistance::getKMeanColors(
    img, n = 2, sample.size = 10000,
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
names(clust_list) <- basename(images)  # match with matrix row names

# Plot the dendrogram
plot(tree, show.tip.label = FALSE, cex = 0.7,
     main = "Dendrogram of image color dissimilarity")

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
  
  if (is.null(cb)) next  # skip if not found
  
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


# ---- Plot dendrogram with images at tips ---- ####
lab_dist <- as.dist(lab_dist_matrix)
hc <- hclust(lab_dist, method = "average")
tree <- as.phylo(hc)
# Create thumbnails
thumbnails <- lapply(images, function(path) {
  image_read(path) |> image_scale("120x120")
})

# Plot tree
plot(tree, show.tip.label = FALSE, main = "Color similarity dendrogram")

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
