###############################################################################
#### CHAPTER 1: COLOR ~ ENVIRONMENT HYPOTHESIS WITH LINEAR DISCRIMINANT ANALYSIS ####
###############################################################################

# ================================================================
# Environment x color-polymorphism association analysis
# Vipera aspis -- POINT-LEVEL version with spatial thinning
#
# Since ~3000 occurrences cover the species range reasonably well,
# environmental clustering is done directly on covariates extracted at
# occurrence points, rather than clustering the full landscape raster
# first. To avoid letting spatially clumped sampling (a known
# GBIF/citizen-science bias) distort the resulting clusters, a spatial
# thinning step precedes each clustering + LDA run, and the whole
# thin -> cluster -> LDA sequence is repeated across many random
# thinning draws to assess how sensitive the result is to any single
# thinning realization.
# ================================================================

# --- 0. Libraries --- ####
library(dplyr)
library(tidyr)
library(sf)
library(terra)
library(farver)
library(cluster)
library(MASS)
library(heplots)
library(vegan)
library(purrr)
library(tibble)
library(usdm)         # vifstep() -- VIF-based collinearity pruning
library(geodata)

set.seed(42)

# ================================================================
# 1. Load occurrence color data
# ================================================================
map_data <- readRDS("map_data.rds")

# ================================================================
# 2. Derive color traits from CIELab (unchanged)
# ================================================================
deg2rad <- function(deg) deg * pi / 180
rad2deg <- function(rad) rad * 180 / pi
WARM_REF_DEG <- 60

trait_vars <- c("lightness_1", "chroma_1", "hue_1_sin", "hue_1_cos",
                "warmth_1", "color_contrast")

color_traits <- map_data %>%
  mutate(
    chroma_1 = sqrt(a_1^2 + b_1^2),
    chroma_2 = sqrt(a_2^2 + b_2^2),
    hue_1_deg = (rad2deg(atan2(b_1, a_1))) %% 360,
    hue_2_deg = (rad2deg(atan2(b_2, a_2))) %% 360,
    hue_1_sin = sin(deg2rad(hue_1_deg)),
    hue_1_cos = cos(deg2rad(hue_1_deg)),
    hue_2_sin = sin(deg2rad(hue_2_deg)),
    hue_2_cos = cos(deg2rad(hue_2_deg)),
    warmth_1 = cos(deg2rad(hue_1_deg - WARM_REF_DEG)) * (chroma_1 / max(chroma_1, na.rm = TRUE)),
    warmth_2 = cos(deg2rad(hue_2_deg - WARM_REF_DEG)) * (chroma_2 / max(chroma_2, na.rm = TRUE)),
    lightness_1 = L_1,
    lightness_2 = L_2
  )

de2000 <- farver::compare_colour(
  from = as.matrix(color_traits[, c("L_1", "a_1", "b_1")]),
  to   = as.matrix(color_traits[, c("L_2", "a_2", "b_2")]),
  from_space = "lab", to_space = "lab", method = "cie2000"
)
color_traits$color_contrast <- diag(as.matrix(de2000))

color_traits <- color_traits %>%
  dplyr::filter(!is.na(latitude), !is.na(longitude)) %>%
  dplyr::select(gbif_id, latitude, longitude, observation_date,
         lightness_1, lightness_2, chroma_1, chroma_2,
         hue_1_deg, hue_2_deg, hue_1_sin, hue_1_cos, hue_2_sin, hue_2_cos,
         warmth_1, warmth_2, color_contrast,
         L_1, a_1, b_1, L_2, a_2, b_2)

occ_sf   <- st_as_sf(color_traits, coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)
occ_vect <- terra::vect(occ_sf)

# ================================================================
# 3. Extract environmental covariates at occurrence points
# ================================================================
# Load layers that were downloaded manually
envroot <- "C:/Users/pdeschepper/OneDrive - Institute of Natural Sciences/Desktop/PERSONAL/Land_dataproducts/"
land_cover_path  <- paste0(envroot, "Corine_landcover_vector_2018/103580/Results/u2018_clc2018_v2020_20u1_raster100m/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif")
# vegetation_path   <- paste0(envroot, "ESA_NDVI_2021/terrascope_download_20260829_234757/WORLDCOVER/ESA_WORLDCOVER_10M_2021_V200/NDVI/ESA_WorldCover_10m_2021_v200_N34E000_NDVI.tif")   
lithology_path     <- paste0(envroot, "EGDI_europe_lithology/GeologicUnitView.gpkg")              

# # Get elevation and worldclim from the geodata package, load them
# dir.create(paste0(envroot, "elevation/"),
#            recursive = TRUE, showWarnings = FALSE)
# dir.create(paste0(envroot, "bioclim/"),
#            recursive = TRUE, showWarnings = FALSE)
# # Download elevation
# elevation <- geodata::elevation_global(
#   res = 2.5,
#   path = paste0(envroot, "elevation/")
# )
# # Download WorldClim bioclimatic variables
# bioclim <- geodata::worldclim_global(
#   var = "bio",
#   res = 2.5,
#   path = paste0(envroot, "bioclim/")
# )
bioclim_path <- paste0(envroot, "bioclim/climate/wc2.1_2.5m") # this contains multiple raster files
bio_files <- list.files(
  bioclim_path,
  pattern = "\\.tif$",
  full.names = TRUE
)
bioclim <- terra::rast(bio_files)
elevation_path <- paste0(envroot, "elevation/elevation/wc2.1_2.5m/wc2.1_2.5m_elev.tif")

extract_raster_at_points <- function(path, points_vect, field_name = NULL) {
  r <- terra::rast(path)
  pts <- terra::project(points_vect, terra::crs(r))
  vals <- terra::extract(r, pts)[, -1, drop = FALSE]  # drop ID column
  if (!is.null(field_name) && ncol(vals) == 1) colnames(vals) <- field_name
  vals
}

extract_vector_at_points <- function(path, points_vect, field) {
  v <- terra::vect(path)
  pts <- terra::project(points_vect, terra::crs(v))
  vals <- terra::extract(v, pts)
    id_col <- names(vals)[1]
    vals <- vals[!duplicated(vals[[id_col]]), ]
    out <- vals[[field]][match(seq_len(length(pts)), vals[[id_col]])]
    setNames(data.frame(out), field)
}

# Extract values
land_cover_vals <- extract_raster_at_points(land_cover_path, occ_vect, "land_cover")
bioclim_vals <- terra::extract(bioclim, occ_vect)[, -1, drop = FALSE]
elevation_vals  <- extract_raster_at_points(elevation_path, occ_vect, "elevation")
# vegetation_vals  <- extract_raster_at_points(vegetation_path, occ_vect, "vegetation")
lithology_vals  <- extract_vector_at_points(lithology_path, occ_vect, "representativelithology_title")
names(lithology_vals) <- "lithology"


point_env_data <- bind_cols(
  color_traits, land_cover_vals, lithology_vals, bioclim_vals, elevation_vals)

bioclim_names    <- grep("bio", names(point_env_data), value = TRUE)
categorical_vars <- c("land_cover", "lithology")

# --- Reduce collinearity among the 19 bioclim variables via VIF-based
#     pruning (usdm::vifstep), ONCE on the full (unthinned) point set,
#     so every thinned replicate below clusters on the SAME retained,
#     named variables. Unlike a PCA axis (a mix of all 19 variables),
#     each retained variable stays directly interpretable -- e.g. a
#     cluster can be described as "high bio1, low bio12" rather than
#     "high PC1". vifstep() iteratively drops the variable with the
#     highest VIF until all remaining variables are below `th`.
#     (vifcor() is a faster, pairwise-correlation-first alternative if
#     you want a more conservative initial pass.)
vif_result <- usdm::vifstep(as.data.frame(point_env_data[, bioclim_names]), th = 5)
bioclim_selected <- as.character(vif_result@results$Variables)
cat("Bioclim variables retained after VIF pruning (", length(bioclim_selected),
    "of", length(bioclim_names), "):\n")
print(bioclim_selected)

continuous_vars <- c("elevation", bioclim_selected)

setwd("C:/Users/pdeschepper/OneDrive - Institute of Natural Sciences/Desktop/PERSONAL/Land_dataproducts/")
saveRDS(point_env_data, "point_env_data.rds")

# ================================================================
# 4. Spatial thinning
# ================================================================
# Simple greedy thinner: shuffle points randomly, then keep a point only
# if it's farther than THIN_KM from every already-kept point. A fresh
# random order each replicate gives a different, equally valid thinned
# subset -- the basis for the robustness check below.
# (For a widely-cited published alternative, see spThin::thin(), which
#  uses a related but not identical stochastic removal algorithm.)
full_dist_km <- as.matrix(st_distance(occ_sf)) / 1000  # computed once, reused by every replicate
full_dist_km <- units::drop_units(full_dist_km)

spatial_thin <- function(idx, dist_mat, thin_km) {
  ord <- sample(idx)
  kept <- integer(0)
  for (i in ord) {
    if (length(kept) == 0 || all(dist_mat[i, kept] >= thin_km)) kept <- c(kept, i)
  }
  kept
}

THIN_KM <- 5      # EDIT: minimum distance (km) between retained points --
# pick this based on something biologically meaningful
# (dispersal distance / independence of local demes)
N_REPS  <- 50    # number of thinning replicates for the robustness check
# (runtime scales with N_REPS; reduce if this is slow)

# ================================================================
# 5. One iteration: thin -> point-level env clustering -> LDA
# ================================================================
run_one_iteration <- function(rep_id, include_subspecies = FALSE) {
  
  keep_idx <- spatial_thin(seq_len(nrow(point_env_data)), full_dist_km, THIN_KM)
  d <- point_env_data[keep_idx, ]
  
  cat_vars_use <- if (include_subspecies) categorical_vars else setdiff(categorical_vars, "subspecies")
  # NOTE: subspecies excluded by default from clustering inputs here too,
  # for the same circularity reason discussed earlier. Set
  # include_subspecies = TRUE to run the sensitivity check.
  
  # continuous_vars already includes elevation/vegetation + the VIF-selected,
  # named bioclim variables (see step 3) -- each is simply scaled within
  # this replicate's thinned subset, no PCA projection needed.
  complete_idx <- complete.cases(d %>% dplyr::select(all_of(cat_vars_use), all_of(continuous_vars)))
  d <- d[complete_idx, ]
  
  clust_input <- d %>%
    dplyr::select(all_of(cat_vars_use), all_of(continuous_vars)) %>%
    mutate(across(all_of(cat_vars_use), ~ droplevels(as.factor(.x)))) %>%
    mutate(across(all_of(continuous_vars), ~ as.numeric(scale(.x)))) %>% 
    na.omit()
  
  clust_matrix <- model.matrix(~ . - 1, data = clust_input %>% dplyr::select(all_of(cat_vars_use))) %>%
    as.data.frame() %>%
    bind_cols(clust_input %>% dplyr::select(all_of(continuous_vars)))
  
  if (nrow(clust_matrix) < 20) return(NULL)  # thinning too aggressive for this replicate
  
  max_k <- min(10, nrow(clust_matrix) - 1)
  sil_widths <- sapply(2:max_k, function(k) {
    km <- kmeans(clust_matrix, centers = k, nstart = 5)
    mean(cluster::silhouette(km$cluster, dist(clust_matrix))[, 3])
  })
  best_k <- (2:max_k)[which.max(sil_widths)]
  km <- kmeans(clust_matrix, centers = best_k, nstart = 10)
  d$env_cluster <- factor(km$cluster)
  
  lda_data <- d %>% dplyr::select(all_of(trait_vars), env_cluster) %>% na.omit()
  if (nlevels(droplevels(lda_data$env_cluster)) < 2) return(NULL)
  lda_data$env_cluster <- droplevels(lda_data$env_cluster)
  
  box_m   <- tryCatch(heplots::boxM(as.matrix(lda_data[, trait_vars]), lda_data$env_cluster),
                      error = function(e) NULL)
  fit_lda <- tryCatch(MASS::lda(env_cluster ~ ., data = lda_data), error = function(e) NULL)
  
  trait_dist <- dist(scale(lda_data[, trait_vars]))
  adonis_res <- tryCatch(
    vegan::adonis2(trait_dist ~ env_cluster, data = lda_data, permutations = 999),
    error = function(e) NULL
  )
  
  list(
    rep = rep_id,
    n_points = nrow(d),
    k = best_k,
    boxM_p = if (!is.null(box_m)) box_m$p.value else NA,
    adonis_R2 = if (!is.null(adonis_res)) adonis_res$R2[1] else NA,
    adonis_p  = if (!is.null(adonis_res)) adonis_res$`Pr(>F)`[1] else NA,
    ld1_loadings = if (!is.null(fit_lda)) fit_lda$scaling[, 1] else NA
  )
}

# ================================================================
# 6. Run all replicates and summarize robustness
# ================================================================

results <- purrr::map(1:N_REPS, run_one_iteration)
results <- purrr::compact(results)
cat(length(results), "of", N_REPS, "replicates completed successfully\n")

summary_df <- purrr::map_dfr(results, ~ tibble(
  rep = .x$rep, n_points = .x$n_points, k = .x$k,
  boxM_p = .x$boxM_p, adonis_R2 = .x$adonis_R2, adonis_p = .x$adonis_p
))

cat("\n--- Robustness summary across", nrow(summary_df), "thinning replicates ---\n")
cat("k chosen:\n"); print(table(summary_df$k))
cat(sprintf("adonis2 R2: median = %.3f (IQR %.3f-%.3f)\n",
            median(summary_df$adonis_R2, na.rm = TRUE),
            quantile(summary_df$adonis_R2, 0.25, na.rm = TRUE),
            quantile(summary_df$adonis_R2, 0.75, na.rm = TRUE)))
cat(sprintf("adonis2 significant (p<0.05) in %d of %d replicates (%.0f%%)\n",
            sum(summary_df$adonis_p < 0.05, na.rm = TRUE), nrow(summary_df),
            100 * mean(summary_df$adonis_p < 0.05, na.rm = TRUE)))

# Stability of which color traits drive LD1 across replicates: a trait
# whose loading flips sign between reps, or averages near zero, isn't a
# robust driver of the environment-color association.
ld1_mat <- purrr::map_dfr(results, function(r) {
  if (length(r$ld1_loadings) == length(trait_vars)) as.list(r$ld1_loadings) else NULL
})
cat("\nLD1 loadings across replicates (mean +/- SD):\n")
print(sapply(ld1_mat, function(x) sprintf("%.2f +/- %.2f", mean(x, na.rm = TRUE), sd(x, na.rm = TRUE))))

saveRDS(list(summary = summary_df, results = results), "thinning_robustness_results.rds")
write.csv(summary_df, "thinning_robustness_summary.csv", row.names = FALSE)

# ================================================================
# 7. CONFIRMATORY: partial Mantel test (color ~ environment | geography)
# ================================================================
# Runs on the full, unthinned point set: the partial correlation already
# controls for geographic distance directly, so it doesn't carry the same
# discretization-driven sampling-density vulnerability the clustering
# step does. Re-run on a thinned subset too if you want full consistency
# with the LDA robustness check above.



# ================================================================
# 7a. VISUALIZATION: LDA & Robustness
# ================================================================


# --- Additional Visualization Libraries --- ####
library(ggplot2)
library(patchwork)
library(ggpubr)

# Set publication-ready theme
theme_set(theme_bw(base_size = 12) + 
            theme(panel.grid = element_blank(),
                  legend.position = "right",
                  plot.title = element_text(face = "bold", size = 13)))

# 1. Select the representative replicate (closest to median R2)
med_r2 <- median(summary_df$adonis_R2, na.rm = TRUE)
best_rep_id <- summary_df$rep[which.min(abs(summary_df$adonis_R2 - med_r2))]

# Re-run single iteration to get model objects for visualization
set.seed(best_rep_id + 42) # match seed logic if applicable
rep_data <- run_one_iteration(best_rep_id)

# Re-extract data for plotting representative LDA
keep_idx <- spatial_thin(seq_len(nrow(point_env_data)), full_dist_km, THIN_KM)
d_rep <- point_env_data[keep_idx, ]
cat_vars_use <- setdiff(categorical_vars, "subspecies")
complete_idx <- complete.cases(d_rep %>% dplyr::select(all_of(cat_vars_use), all_of(continuous_vars)))
d_rep <- d_rep[complete_idx, ]

clust_input <- d_rep %>%
  dplyr::select(all_of(cat_vars_use), all_of(continuous_vars)) %>%
  mutate(across(all_of(cat_vars_use), ~ droplevels(as.factor(.x)))) %>%
  mutate(across(all_of(continuous_vars), ~ as.numeric(scale(.x))))

clust_matrix <- model.matrix(~ . - 1, data = clust_input %>% dplyr::select(all_of(cat_vars_use))) %>%
  as.data.frame() %>%
  bind_cols(clust_input %>% dplyr::select(all_of(continuous_vars)))

km_rep <- kmeans(clust_matrix, centers = rep_data$k, nstart = 10)
d_rep$env_cluster <- factor(km_rep$cluster)

lda_data <- d_rep %>% dplyr::select(all_of(trait_vars), env_cluster) %>% na.omit()
lda_fit <- MASS::lda(env_cluster ~ ., data = lda_data)
lda_pred <- predict(lda_fit)

# Combine LDA scores into a plotting frame
lda_scores <- as.data.frame(lda_pred$x) %>%
  bind_cols(env_cluster = lda_data$env_cluster)

# --- Plot A: LDA Ordination (LD1 vs LD2) ---
if (ncol(lda_pred$x) >= 2) {
  p_lda_ord <- ggplot(lda_scores, aes(x = LD1, y = LD2, color = env_cluster, fill = env_cluster)) +
    stat_ellipse(geom = "polygon", alpha = 0.2, level = 0.95) +
    geom_point(alpha = 0.7, size = 2) +
    scale_color_viridis_d(name = "Env Cluster") +
    scale_fill_viridis_d(name = "Env Cluster") +
    labs(title = "A. Linear Discriminant Analysis",
         subtitle = paste("Representative Replicate (k =", rep_data$k, ")"),
         x = "LD1 Axis", y = "LD2 Axis")
} else {
  # Fallback density plot if only 2 clusters exist (1 LD axis)
  p_lda_ord <- ggplot(lda_scores, aes(x = LD1, fill = env_cluster)) +
    geom_density(alpha = 0.5) +
    scale_fill_viridis_d(name = "Env Cluster") +
    labs(title = "A. Discriminant Axis 1 Density",
         subtitle = paste("Representative Replicate (k =", rep_data$k, ")"),
         x = "LD1 Axis", y = "Density")
}

# --- Plot B: LD1 Loading Stability Across Replicates ---
ld1_df <- as.data.frame(ld1_mat) %>%
  pivot_longer(cols = everything(), names_to = "Trait", values_to = "Loading")

p_ld1_loadings <- ggplot(ld1_df, aes(x = reorder(Trait, abs(Loading), FUN = median), y = Loading)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_boxplot(fill = "skyblue", alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.15, alpha = 0.5, size = 1.5, color = "darkblue") +
  coord_flip() +
  labs(title = "B. Trait Loadings Stability (LD1)",
       subtitle = paste("Across", length(results), "Spatial Thinning Replicates"),
       x = "Color Traits", y = "LD1 Loading Coefficient")

# Combine 7a plots
p_7a <- p_lda_ord + p_ld1_loadings + plot_layout(ncol = 2)
print(p_7a)
ggsave("figure_7a_lda_analysis.png", p_7a, width = 11, height = 5, dpi = 300)

# ================================================================
# 7b. VISUALIZATION: Environmental Clusters in Geographic Space
# ================================================================

library(ggplot2)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(viridis)


# --- 1. Load European Spatial Basemaps ---
# Download land polygons and country boundaries for Western Europe
world_land <- ne_countries(scale = "medium", returnclass = "sf")

# Define spatial extent bounding box based on your occurrence data
# (Buffer bounds by ~1 degree for better visualization)
lon_min <- min(point_env_data$longitude, na.rm = TRUE) - 1
lon_max <- max(point_env_data$longitude, na.rm = TRUE) + 1
lat_min <- min(point_env_data$latitude, na.rm = TRUE) - 1
lat_max <- max(point_env_data$latitude, na.rm = TRUE) + 1

# --- 2. Extract Data for Mapping ---
# Option A: Plot representative thinned replicate (from Step 7a)
# (Uses `d_rep` containing `env_cluster` and coordinates)
rep_sf <- st_as_sf(d_rep, coords = c("longitude", "latitude"), crs = 4326)

# --- 3. Build Spatial Map ---
p_env_map <- ggplot() +
  # Base map: Country boundaries
  geom_sf(data = world_land, fill = "gray92", color = "gray70", linewidth = 0.3) +
  # Environmental Clusters
  geom_sf(data = rep_sf, aes(color = env_cluster), size = 2.2, alpha = 0.85) +
  # Zoom to species geographic distribution extent
  coord_sf(xlim = c(lon_min, lon_max), 
           ylim = c(lat_min, lat_max), 
           expand = FALSE) +
  # Styling and Palette
  scale_color_viridis_d(name = "Env Cluster", option = "D") +
  theme_bw(base_size = 12) +
  theme(
    panel.background = element_rect(fill = "aliceblue"), # Ocean color
    panel.grid.major = element_line(color = "gray85", linetype = "dotted"),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.box.background = element_rect(fill = "white", color = "gray80"),
    plot.title = element_text(face = "bold", size = 13)
  ) +
  labs(
    title = "Geographic Distribution of Environmental Clusters",
    subtitle = paste0("Vipera aspis — Representative Thinned Replicate (k = ", rep_data$k, ")"),
    x = "Longitude", 
    y = "Latitude"
  )

print(p_env_map)


###############################################################################
#### CHAPTER 2: NONRANDOM DISTRIBUTION OF COLOR ####
###############################################################################

# --- 1. Prepare Trait, Spatial, and Environmental Matrices ---
# Response: CIELAB color parameters for both background and pattern
Y_color <- point_env_data %>%
  dplyr::select(L_1, a_1, b_1, L_2, a_2, b_2) %>%
  as.matrix()

# Coordinates
coords <- point_env_data %>%
  dplyr::select(longitude, latitude) %>%
  as.matrix()

# Environment: Continuous climate/elevation + dummy-encoded categorical variables
E_env <- point_env_data %>%
  dplyr::select(starts_with("wc2.1_2.5m_bio_"), elevation, land_cover, lithology) %>%
  # Model matrix automatically handles factor dummy encoding (land_cover, lithology)
  model.matrix(~ . - 1, data = .) %>%
  scale()

### 2.1 Partial Mantel Test ###
mantel_data <- point_env_data %>%
  dplyr::select(gbif_id, latitude, longitude, L_1, a_1, b_1,
                all_of(continuous_vars), land_cover, lithology) %>%
  na.omit()

color_dist <- as.dist(farver::compare_colour(
  from = as.matrix(mantel_data[, c("L_1", "a_1", "b_1")]),
  to   = as.matrix(mantel_data[, c("L_1", "a_1", "b_1")]),
  from_space = "lab", to_space = "lab", method = "cie2000"
))

env_dist <- cluster::daisy(
  mantel_data %>%
    dplyr::select(all_of(continuous_vars), land_cover, lithology) %>%
    mutate(across(c(land_cover, lithology), as.factor)),
  metric = "gower"
)

pts <- st_as_sf(mantel_data, coords = c("longitude", "latitude"), crs = 4326)
geo_dist <- as.dist(units::drop_units(st_distance(pts)) / 1000)
attr(geo_dist, "Size") <- nrow(mantel_data)

partial_mantel <- vegan::mantel.partial(
  color_dist, env_dist, geo_dist, method = "spearman", permutations = 10
)
print(partial_mantel)

cat("\nDone. Key outputs: point_env_data.rds, thinning_robustness_summary.csv,",
    "thinning_robustness_results.rds, partial_mantel\n")

# Visualization of mantel results

# --- Plot A: Matrix Distance Correlations ---
# Subsample matrices for visual scannability (Mantel plots saturate with high N)
set.seed(42)
sample_n <- min(1000, length(color_dist))
sub_idx <- sample(seq_along(color_dist), sample_n)

df_mantel <- data.frame(
  Color_Dist = as.vector(color_dist)[sub_idx],
  Env_Dist = as.vector(env_dist)[sub_idx],
  Geo_Dist = as.vector(geo_dist)[sub_idx]
)

# 1. Color vs Environment
p_mantel_env <- ggplot(df_mantel, aes(x = Env_Dist, y = Color_Dist)) +
  geom_point(alpha = 0.2, size = 1, color = "darkslategrey") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = "A. Color vs. Environmental Distance",
       subtitle = paste("Partial Mantel r =", round(partial_mantel$statistic, 3), 
                        "| p =", partial_mantel$signif),
       x = "Environmental Distance (Gower)",
       y = "Color Distance (CIEDE2000)")

# 2. Color vs Geography
p_mantel_geo <- ggplot(df_mantel, aes(x = Geo_Dist, y = Color_Dist)) +
  geom_point(alpha = 0.2, size = 1, color = "darkslategrey") +
  geom_smooth(method = "lm", color = "royalblue", se = TRUE) +
  labs(title = "B. Color vs. Geographic Distance",
       x = "Geographic Distance (km)",
       y = "Color Distance (CIEDE2000)")

# --- Plot B: Permutation Test Null Distribution ---
null_dist <- data.frame(r = partial_mantel$perm)

p_mantel_null <- ggplot(null_dist, aes(x = r)) +
  geom_histogram(fill = "gray80", color = "gray40", bins = 20) +
  geom_vline(xintercept = partial_mantel$statistic, color = "red", linetype = "dashed", size = 1) +
  labs(title = "C. Partial Mantel Permutation Test",
       subtitle = "Observed Statistic (Red) vs. Permutation Null",
       x = "Mantel Correlation Coefficient (r)",
       y = "Frequency")

# Combine 7b plots
p_7b <- (p_mantel_env | p_mantel_geo) / p_mantel_null + plot_layout(heights = c(2, 1.2))
print(p_7b)
ggsave("figure_7b_mantel_test.png", p_7b, width = 10, height = 8, dpi = 300)


### 2.2 Fit Generalized Additive Model on Background Lightness ####
# Uses a Gaussian Process smooth s(longitude, latitude) alongside environmental main terms
gam_lightness <- gam(
  color_contrast ~ s(longitude, latitude, bs = "gp", k = 50) + 
    s(wc2.1_2.5m_bio_1) + 
    s(wc2.1_2.5m_bio_12) + 
    lithology +
    land_cover +
    s(elevation),
  data = point_env_data,
  method = "REML"
)

summary(gam_lightness)

# Visualize pure spatial smooth surface
plot(gam_lightness, select = 1, scheme = 2, main = "Pure Spatial Effect on Lightness 1")

##################################

library(mgcv)
library(ggplot2)
library(sf)
library(rnaturalearth)

# Helper function to get mode of categorical variables
get_mode <- function(x) {
  ux <- na.omit(unique(x))
  ux[which.max(tabulate(match(x, ux)))]
}

# 1. Fetch country borders
countries <- ne_countries(scale = "medium", returnclass = "sf")

# 2. Build 2D grid over sampling coordinates
grid_df <- expand.grid(
  longitude = seq(min(clean_data$longitude), max(clean_data$longitude), length.out = 100),
  latitude  = seq(min(clean_data$latitude), max(clean_data$latitude), length.out = 100)
)

# 3. Fill ALL non-spatial model variables (both continuous and factor/character)
# This includes all variables present in gam_lightness formula
grid_df$wc2.1_2.5m_bio_1  <- median(clean_data$wc2.1_2.5m_bio_1, na.rm = TRUE)
grid_df$wc2.1_2.5m_bio_12 <- median(clean_data$wc2.1_2.5m_bio_12, na.rm = TRUE)
grid_df$elevation         <- median(clean_data$elevation, na.rm = TRUE)
grid_df$land_cover        <- get_mode(clean_data$land_cover)
grid_df$lithology         <- get_mode(clean_data$lithology)

# 4. Predict partial terms directly from gam_lightness
terms_mat <- predict(gam_lightness, newdata = grid_df, type = "terms")

# Identify and extract s(longitude,latitude)
spatial_col <- grep("longitude,latitude", colnames(terms_mat), value = TRUE)
grid_df$spatial_effect <- terms_mat[, spatial_col]

# 5. Mask grid points that are too far from sampling points
far_mask <- exclude.too.far(
  g1 = grid_df$longitude, 
  g2 = grid_df$latitude, 
  d1 = clean_data$longitude, 
  d2 = clean_data$latitude, 
  dist = 0.1
)
spatial_df <- grid_df[!far_mask, ]

# Crop bounds
lon_range <- range(spatial_df$longitude)
lat_range <- range(spatial_df$latitude)

# 6. Plot with ggplot2
ggplot() +
  # Continuous spatial smooth surface
  geom_raster(
    data = spatial_df, 
    aes(x = longitude, y = latitude, fill = spatial_effect), 
    interpolate = TRUE
  ) +
  
  # Country borders overlay
  geom_sf(data = countries, fill = NA, color = "black", linewidth = 0.4) +
  
  # Sampling observation points
  geom_point(
    data = clean_data, 
    aes(x = longitude, y = latitude), 
    color = "black", alpha = 0.3, size = 0.8
  ) +
  
  # Palette
  scale_fill_viridis_c(name = "Partial Effect\n(Lightness 1)", option = "plasma") +
  
  # Crop view
  coord_sf(
    xlim = lon_range, 
    ylim = lat_range, 
    expand = FALSE
  ) +
  
  labs(
    title = "Pure Spatial Effect on Background Lightness",
    subtitle = "Partial smooth s(longitude, latitude) controlling for environmental factors",
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_bw() +
  theme(
    panel.grid.major = element_line(color = "gray90", linetype = "dashed")
  )
