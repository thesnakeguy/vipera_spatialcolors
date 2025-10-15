library(recolorize)

# 1. Set the path to your extracted images
img_dir <- "C:/Users/pdeschepper/Desktop/PERSONAL/DeepLearning/ImageSegmentation/Snakes_ImageSegmentation_keras/Vipera_segmentation_test_dataset/Extracted_snakes/"

# Get a list of all the PNG files
img_files <- list.files(img_dir, pattern = "*.png$", full.names = TRUE)

# 2. Analyze a single image as an example
# Let's process the first image in the list
example_img_path <- img_files[sample(length(img_files), 1)]

# Load the image using readImage. It's crucial to set `alpha = TRUE`
# to ignore the transparent background during color analysis.
img_to_recolor <- readImage(example_img_path)

# Run the main recolorize function to find the N most dominant colors.
# Let's find the 5 main colors (e.g., dark background, light background, spots, etc.)
# `plotting = TRUE` will show you the result visually.
recolor_result <- recolorize(img_to_recolor, n = 5, plotting = TRUE)

# 3. Inspect the results
# You can see the identified color centers (in RGB) and their proportions
print("Color centers (RGB):")
print(recolor_result$centers)

print("Proportion of each color:")
print(recolor_result$sizes)

