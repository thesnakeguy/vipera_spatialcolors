# ================================================================
# Snake color map: join color clusters + class coverage + GBIF coords,
# average colors (in CIELab space) to one marker per GBIF observation,
# render as a clustered leaflet map with two-color pie markers
# (top half = dominant color, bottom half = secondary color)
# ================================================================

# --- 1. Libraries --- ####
library(dplyr)
library(tidyr)
library(stringr)
library(leaflet)
library(base64enc)
library(purrr)

# --- 2. File paths -- ####
basedir <- "C:/Users/pdeschepper/OneDrive - Institute of Natural Sciences/Desktop/PERSONAL/DeepLearning/vipera_spatialcolors/Gbif_sourceimages/"
color_csv    <- paste0(basedir,"Extracted_snakes_pytorch/ColorExtraction_results/snake_color_clusters.csv")         
coverage_csv <- paste0(basedir,"Extracted_snakes_pytorch/class_coverage.csv")
gbif_csv     <- paste0(basedir,"Vipera_aspis_gbif_metadata.csv")

# Minimum class_coverage (%) a photo must have to be trusted for color
# extraction. Photos below this are dropped before averaging, so a
# low-coverage photo (snake barely visible / mostly background/other class)
# can't drag an observation's averaged color off.
COVERAGE_THRESHOLD <- 10

colors   <- read.csv(color_csv, stringsAsFactors = FALSE)
coverage <- read.csv(coverage_csv, stringsAsFactors = FALSE)
gbif     <- read.csv(gbif_csv, stringsAsFactors = FALSE)

# --- 3. Parse GBIF metadata: one row per photo_id (link table only) --- ####
# photo_ids arrives as a Python-list-formatted string, e.g.
# "['6147507750_1', '6147507750_2']" -- extract each quoted id and unnest.
# This stays a per-photo LINK table so we know which photos belong to
# which observation; the final map data is aggregated back to one row
# per gbif_id in step 7.
gbif_long <- gbif %>%
  mutate(photo_id_list = stringr::str_extract_all(photo_ids, "'([^']+)'")) %>%
  tidyr::unnest(photo_id_list) %>%
  mutate(photo_id = stringr::str_remove_all(photo_id_list, "'")) %>%
  select(gbif_id, latitude, longitude, observation_date, photo_id) %>%
  distinct()

# --- 4. Clean class coverage: strip file extension to match photo_id --- ####
coverage_clean <- coverage %>%
  mutate(photo_id = stringr::str_remove(photo_ID, "\\.[A-Za-z]+$")) %>%
  select(photo_id, class_coverage)

# --- 5. Pivot color clusters to one row per photo --- ####
# Keep L, a, b (not hex) per rank -- we need Lab coordinates to average
# perceptually correctly; hex is only computed after averaging, in step 7.
colors_wide <- colors %>%
  group_by(photo_id) %>%
  arrange(desc(proportion), .by_group = TRUE) %>%
  mutate(color_rank = row_number()) %>%
  ungroup() %>%
  filter(color_rank <= 2) %>%
  select(photo_id, color_rank, proportion, L, a, b) %>%
  tidyr::pivot_wider(
    names_from = color_rank,
    values_from = c(proportion, L, a, b),
    names_glue = "{.value}_{color_rank}"
  )

# --- 6. Join to photo level --- ####
photo_level <- colors_wide %>%
  inner_join(coverage_clean, by = "photo_id") %>%
  inner_join(gbif_long, by = "photo_id")

# Diagnostics: report photos that failed to match, so you can spot check
unmatched_coverage <- anti_join(colors_wide, coverage_clean, by = "photo_id")
unmatched_gbif      <- anti_join(colors_wide, gbif_long, by = "photo_id")
if (nrow(unmatched_coverage) > 0) {
  cat(nrow(unmatched_coverage), "photo_id(s) had no match in class_coverage.csv\n")
}
if (nrow(unmatched_gbif) > 0) {
  cat(nrow(unmatched_gbif), "photo_id(s) had no match in GBIF metadata\n")
}

# --- 7. Aggregate to one row per GBIF observation --- ####
# Colors are averaged in CIELab space (perceptually meaningful, consistent
# with the rest of the pipeline) rather than averaging hex/RGB directly.
# Each photo's contribution to its rank's average is weighted by that
# rank's proportion within the photo, so a color that barely registered
# in one photo doesn't pull the average as hard as a near-50/50 split.
gbif_level <- photo_level %>%
  group_by(gbif_id, latitude, longitude, observation_date) %>%
  summarise(
    n_photos     = n(),
    L_1 = weighted.mean(L_1, w = proportion_1, na.rm = TRUE),
    a_1 = weighted.mean(a_1, w = proportion_1, na.rm = TRUE),
    b_1 = weighted.mean(b_1, w = proportion_1, na.rm = TRUE),
    L_2 = weighted.mean(L_2, w = proportion_2, na.rm = TRUE),
    a_2 = weighted.mean(a_2, w = proportion_2, na.rm = TRUE),
    b_2 = weighted.mean(b_2, w = proportion_2, na.rm = TRUE),
    proportion_1   = mean(proportion_1, na.rm = TRUE),
    proportion_2   = mean(proportion_2, na.rm = TRUE),
    class_coverage = mean(class_coverage, na.rm = TRUE),  # simple mean across photos
    .groups = "drop"
  )

# Convert averaged Lab back to sRGB hex, with gamut clamping
lab_to_hex <- function(L, a, b) {
  m <- cbind(L, a, b)
  rgb_vals <- grDevices::convertColor(m, from = "Lab", to = "sRGB", clip = TRUE)
  rgb_vals[rgb_vals < 0] <- 0
  rgb_vals[rgb_vals > 1] <- 1
  grDevices::rgb(rgb_vals[, 1], rgb_vals[, 2], rgb_vals[, 3])
}

map_data <- gbif_level %>%
  mutate(
    hex_1 = lab_to_hex(L_1, a_1, b_1),
    hex_2 = lab_to_hex(L_2, a_2, b_2)
  ) %>%
  filter(!is.na(latitude), !is.na(longitude), !is.na(hex_1), !is.na(hex_2))

cat(nrow(map_data), "GBIF observations mapped (from",
    n_distinct(photo_level$photo_id), "photos).\n")

# --- 8. Identify the background and pattern color --- ####
# This function returns the a vector with the darker hex code first
identify_darker_color <- function(hex_1, hex_2) {
  # Convert hex codes to RGB matrices (values 0-255)
  rgb_1 <- col2rgb(hex_1)
  rgb_2 <- col2rgb(hex_2)
  
  # Standard weights for perceived human brightness
  weights <- c(0.299, 0.587, 0.114)
  
  # Calculate luminance (lower = darker)
  lum_1 <- sum(rgb_1 * weights)
  lum_2 <- sum(rgb_2 * weights)
  
  # Compare and return the result
  if (lum_1 < lum_2) {
    return(c(hex_1, hex_2))
  } else {
    return(c(hex_2, hex_1))
  }
}

# Example usage:
identify_darker_color("#4A90E2", "#1C3D5A")


# --- 9. Build a two-color (top/bottom) SVG marker icon per observation --- ####
make_pie_icon <- function(hex1, hex2, size = 28) {
  r  <- size / 2 - 1
  cx <- size / 2
  cy <- size / 2
  svg <- sprintf(
    '<svg xmlns="http://www.w3.org/2000/svg" width="%1$d" height="%1$d" viewBox="0 0 %1$d %1$d">
       <defs><clipPath id="clip"><circle cx="%2$d" cy="%2$d" r="%3$d"/></clipPath></defs>
       <g clip-path="url(#clip)">
         <rect x="0" y="0" width="%1$d" height="%2$d" fill="%4$s"/>
         <rect x="0" y="%2$d" width="%1$d" height="%2$d" fill="%5$s"/>
       </g>
       <circle cx="%2$d" cy="%2$d" r="%3$d" fill="none" stroke="#333333" stroke-width="1"/>
     </svg>',
    size, cy, r, hex1, hex2
  )
  base64enc::dataURI(charToRaw(svg), mime = "image/svg+xml", encoding = "base64")
}

# Apply the make_pie_icon and identify_darker_color functions
map_data <- map_data %>%
  mutate(
    icon_uri = pmap_chr(list(hex_1, hex_2), function(h1, h2) {
      colors <- identify_darker_color(h1, h2)
      make_pie_icon(colors[1], colors[2])
    })
  )

icon_size <- 28
pie_icons <- leaflet::icons(
  iconUrl    = map_data$icon_uri,
  iconWidth  = icon_size,
  iconHeight = icon_size
)

# --- 10. Render leaflet map --- ####
# markerClusterOptions() gives the "cluster when zoomed out, split into
# individuals when zoomed in" behavior automatically (Leaflet.markercluster
# under the hood). Since we're now one marker per observation (not per
# photo), overlapping markers should be rarer, but spiderfy still helps
# for co-located observations.
snake_map <- leaflet(map_data) %>%
  addProviderTiles(providers$OpenStreetMap.Mapnik) %>%
  addMarkers(
    lng = ~longitude, lat = ~latitude,
    icon = pie_icons,
    options = markerOptions(riseOnHover = TRUE),
    clusterOptions = markerClusterOptions(spiderfyOnMaxZoom = TRUE,
                                          maxClusterRadius = 30,
                                          disableClusteringAtZoom = 9
                                          ),
    popup = ~sprintf(
      "<b>GBIF ID:</b> <a href='%s' target='_blank'>%s</a><br>
   <b>Photos averaged:</b> %d<br>
   <b>Dominant color:</b> <span style='color:%s'>&#9632;</span> %s (%.0f%%)<br>
   <b>Secondary color:</b> <span style='color:%s'>&#9632;</span> %s (%.0f%%)<br>
   <b>Class coverage:</b> %.1f%%<br>
   <b>Observation date:</b> %s",
      paste0("https://www.gbif.org/occurrence/", gbif_id), gbif_id, n_photos,
      hex_1, hex_1, proportion_1 * 100,
      hex_2, hex_2, proportion_2 * 100,
      class_coverage, observation_date
    )
  ) %>%
  addControl(
    html = "<div style='background:white;padding:6px 10px;border-radius:4px;font-size:12px;'>
              Marker: top half = dominant color, bottom half = secondary color<br>
              (averaged across all photos of the observation)
            </div>",
    position = "bottomleft"
  )

snake_map
# htmlwidgets::saveWidget(snake_map, "snake_color_map.html", selfcontained = TRUE)