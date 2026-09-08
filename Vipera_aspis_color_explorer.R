# ================================================================
# Shiny app: interactive Vipera aspis color explorer
# Filters by month + class coverage, re-averaging colors per
# GBIF observation reactively (not just show/hide of markers).
# ================================================================

library(shiny)
library(leaflet)
library(dplyr)
library(tidyr)
library(stringr)
library(purrr)
library(lubridate)
library(base64enc)
library(plotly)
library(DT)

# ---------------------------------------------------------------
# 1. Load & prepare data (runs once at app startup)
# ---------------------------------------------------------------
basedir <- "C:/Users/pdeschepper/OneDrive - Institute of Natural Sciences/Desktop/PERSONAL/DeepLearning/vipera_spatialcolors/Gbif_sourceimages/"
color_csv    <- paste0(basedir, "Extracted_snakes_pytorch/ColorExtraction_results/snake_color_clusters.csv")
coverage_csv <- paste0(basedir, "Extracted_snakes_pytorch/class_coverage.csv")
gbif_csv     <- paste0(basedir, "Vipera_aspis_gbif_metadata.csv")

colors_raw <- read.csv(color_csv, stringsAsFactors = FALSE)
coverage   <- read.csv(coverage_csv, stringsAsFactors = FALSE)
gbif       <- read.csv(gbif_csv, stringsAsFactors = FALSE)

# --- Parse GBIF metadata: one row per photo_id (link table) ---
# observation_date arrives as either "YYYY-MM-DDTHH:MM[:SS]" or just
# "YYYY-MM-DD" -- ymd_hms with truncated = 3 handles both.
gbif_long <- gbif %>%
  mutate(
    photo_id_list = stringr::str_extract_all(photo_ids, "'([^']+)'"),
    date_parsed   = lubridate::ymd_hms(stringr::str_replace(observation_date, "T", " "),
                                       truncated = 3, quiet = TRUE)
  ) %>%
  tidyr::unnest(photo_id_list) %>%
  mutate(photo_id = stringr::str_remove_all(photo_id_list, "'")) %>%
  select(gbif_id, latitude, longitude, observation_date, date_parsed, photo_id) %>%
  distinct()

n_bad_dates <- sum(is.na(gbif_long$date_parsed))
if (n_bad_dates > 0) cat(n_bad_dates, "photo(s) had an unparseable observation_date\n")

# --- Photo URL gallery lookup, keyed by gbif_id (independent of filters) ---
gbif_photo_urls <- gbif %>%
  mutate(photo_url_list = stringr::str_extract_all(photos, "'([^']+)'")) %>%
  mutate(photo_url_list = purrr::map(photo_url_list, ~ stringr::str_remove_all(.x, "'"))) %>%
  select(gbif_id, photo_url_list)
photo_urls_by_id <- setNames(gbif_photo_urls$photo_url_list, as.character(gbif_photo_urls$gbif_id))

# --- Clean class coverage: strip extension to match photo_id ---
coverage_clean <- coverage %>%
  mutate(photo_id = stringr::str_remove(photo_ID, "\\.[A-Za-z]+$")) %>%
  select(photo_id, class_coverage)

# --- Pivot color clusters to one row per photo (L,a,b kept for averaging) ---
colors_wide <- colors_raw %>%
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

# --- Join to photo level. NOT filtered by coverage threshold here --
#     that filter is applied reactively inside the app, since changing
#     it changes which photos feed into each observation's color average.
photo_level_all <- colors_wide %>%
  inner_join(coverage_clean, by = "photo_id") %>%
  inner_join(gbif_long, by = "photo_id") %>%
  mutate(month = lubridate::month(date_parsed, label = TRUE, abbr = FALSE))  # ordered factor, full names

unmatched_coverage <- anti_join(colors_wide, coverage_clean, by = "photo_id")
unmatched_gbif      <- anti_join(colors_wide, gbif_long, by = "photo_id")
if (nrow(unmatched_coverage) > 0) cat(nrow(unmatched_coverage), "photo_id(s) had no match in class_coverage.csv\n")
if (nrow(unmatched_gbif) > 0)      cat(nrow(unmatched_gbif), "photo_id(s) had no match in GBIF metadata\n")

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
lab_to_hex <- function(L, a, b) {
  m <- cbind(L, a, b)
  rgb_vals <- grDevices::convertColor(m, from = "Lab", to = "sRGB", clip = TRUE)
  rgb_vals[rgb_vals < 0] <- 0
  rgb_vals[rgb_vals > 1] <- 1
  grDevices::rgb(rgb_vals[, 1], rgb_vals[, 2], rgb_vals[, 3])
}

identify_darker_color <- function(hex_1, hex_2) {
  rgb_1 <- grDevices::col2rgb(hex_1)
  rgb_2 <- grDevices::col2rgb(hex_2)
  weights <- c(0.299, 0.587, 0.114)
  lum_1 <- sum(rgb_1 * weights)
  lum_2 <- sum(rgb_2 * weights)
  if (lum_1 < lum_2) c(hex_1, hex_2) else c(hex_2, hex_1)
}

make_pie_icon <- function(hex1, hex2, size = 28) {
  r  <- size / 2 - 1
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

# Aggregates an already-filtered photo-level data frame to one row per
# GBIF observation. Re-run reactively whenever the filters change, since
# which photos are included changes the average.
build_map_data <- function(photo_level) {
  if (nrow(photo_level) == 0) return(photo_level)
  
  gbif_level <- photo_level %>%
    group_by(gbif_id, latitude, longitude, observation_date, month) %>%
    summarise(
      n_photos = n(),
      L_1 = weighted.mean(L_1, w = proportion_1, na.rm = TRUE),
      a_1 = weighted.mean(a_1, w = proportion_1, na.rm = TRUE),
      b_1 = weighted.mean(b_1, w = proportion_1, na.rm = TRUE),
      L_2 = weighted.mean(L_2, w = proportion_2, na.rm = TRUE),
      a_2 = weighted.mean(a_2, w = proportion_2, na.rm = TRUE),
      b_2 = weighted.mean(b_2, w = proportion_2, na.rm = TRUE),
      proportion_1   = mean(proportion_1, na.rm = TRUE),
      proportion_2   = mean(proportion_2, na.rm = TRUE),
      class_coverage = mean(class_coverage, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      hex_1 = lab_to_hex(L_1, a_1, b_1),
      hex_2 = lab_to_hex(L_2, a_2, b_2)
    ) %>%
    filter(!is.na(latitude), !is.na(longitude), !is.na(hex_1), !is.na(hex_2))
  
  if (nrow(gbif_level) == 0) return(gbif_level)
  
  gbif_level %>%
    mutate(
      dark_light = purrr::map2(hex_1, hex_2, identify_darker_color),
      hex_dark   = purrr::map_chr(dark_light, 1),
      hex_light  = purrr::map_chr(dark_light, 2),
      icon_uri   = purrr::map2_chr(hex_dark, hex_light, make_pie_icon)
    ) %>%
    select(-dark_light)
}

month_choices <- month.name

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
ui <- navbarPage(
  title = "Vipera aspis color explorer",
  
  tabPanel("Map",
           sidebarLayout(
             sidebarPanel(
               width = 3,
               sliderInput("coverage_threshold", "Minimum class coverage (%)",
                           min = 0, max = 100, value = 10, step = 1),
               checkboxGroupInput("months", "Months",
                                  choices = month_choices, selected = month_choices),
               actionButton("select_all_months", "Select all", class = "btn-sm"),
               actionButton("select_no_months", "Clear all", class = "btn-sm"),
               hr(),
               htmlOutput("summary_box"),
               hr(),
               h5("Selected observation"),
               uiOutput("photo_gallery")
             ),
             mainPanel(
               width = 9,
               leafletOutput("snake_map", height = 700)
             )
           )
  ),
  
  tabPanel("Color space",
           plotlyOutput("color_scatter", height = 650)
  ),
  
  tabPanel("Seasonal trends",
           plotOutput("month_trend", height = 650)
  ),
  
  tabPanel("Data",
           DTOutput("data_table")
  )
)

# ---------------------------------------------------------------
# Server
# ---------------------------------------------------------------
server <- function(input, output, session) {
  
  observeEvent(input$select_all_months, {
    updateCheckboxGroupInput(session, "months", selected = month_choices)
  })
  observeEvent(input$select_no_months, {
    updateCheckboxGroupInput(session, "months", selected = character(0))
  })
  
  # Debounce the coverage slider so dragging it doesn't trigger a full
  # re-aggregation on every intermediate pixel value.
  coverage_threshold_d <- reactive(input$coverage_threshold) %>% debounce(300)
  
  filtered_photo_level <- reactive({
    req(input$months)
    photo_level_all %>%
      filter(class_coverage > coverage_threshold_d(),
             as.character(month) %in% input$months)
  })
  
  map_data <- reactive({
    build_map_data(filtered_photo_level())
  })
  
  output$summary_box <- renderUI({
    md <- map_data()
    HTML(sprintf(
      "<b>%d</b> observations shown<br><b>%d</b> photos contributing",
      nrow(md), if (nrow(md) > 0) sum(md$n_photos) else 0
    ))
  })
  
  # --- Map: draw base tiles once, update markers via leafletProxy ---
  output$snake_map <- renderLeaflet({
    leaflet() %>%
      addProviderTiles(providers$OpenStreetMap.Mapnik) %>%
      setView(lng = 4, lat = 46, zoom = 6)
  })
  
  observe({
    md <- map_data()
    proxy <- leafletProxy("snake_map") %>% clearMarkers() %>% clearMarkerClusters()
    if (nrow(md) == 0) return(invisible(NULL))
    
    icons <- leaflet::icons(iconUrl = md$icon_uri, iconWidth = 28, iconHeight = 28)
    
    proxy %>% addMarkers(
      data = md,
      lng = ~longitude, lat = ~latitude,
      icon = icons,
      layerId = ~as.character(gbif_id),
      options = markerOptions(riseOnHover = TRUE),
      clusterOptions = markerClusterOptions(
        spiderfyOnMaxZoom = TRUE, maxClusterRadius = 40, disableClusteringAtZoom = 12
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
    )
  })
  
  # --- Click-to-view photo gallery (pulls real images from GBIF/iNat/observation.org) ---
  output$photo_gallery <- renderUI({
    click <- input$snake_map_marker_click
    if (is.null(click)) return(tags$p("Click a marker to see its photos.", style = "color:#888;"))
    
    urls <- photo_urls_by_id[[click$id]]
    if (is.null(urls) || length(urls) == 0) return(tags$p("No photos found for this observation."))
    
    tagList(lapply(urls, function(u)
      tags$img(src = u, style = "width:100%; margin-bottom:6px; border-radius:4px;")
    ))
  })
  
  # --- Color space scatter: dominant color of each observation, a*/b* ---
  output$color_scatter <- renderPlotly({
    md <- map_data()
    if (nrow(md) == 0) return(NULL)
    
    plot_ly(
      md, x = ~a_1, y = ~b_1,
      type = "scatter", mode = "markers",
      marker = list(color = ~hex_1, size = ~pmax(proportion_1 * 40, 6),
                    line = list(color = "#333333", width = 1)),
      text = ~sprintf("GBIF %s<br>%d photos<br>%s", gbif_id, n_photos, observation_date),
      hoverinfo = "text"
    ) %>%
      layout(
        title = "Dominant color of each observation in a*/b* space",
        xaxis = list(title = "a* (green-red)"),
        yaxis = list(title = "b* (blue-yellow)")
      )
  })
  
  # --- Seasonal trend: uses the coverage filter but ignores the month
  #     checkbox filter, so the full seasonal pattern is always visible
  #     to help decide which months to select on the Map tab. ---
  output$month_trend <- renderPlot({
    fp <- photo_level_all %>% filter(class_coverage > coverage_threshold_d())
    md_full <- build_map_data(fp)
    if (nrow(md_full) == 0) return(NULL)
    
    summary_df <- md_full %>%
      group_by(month, .drop = FALSE) %>%
      summarise(n = n(), mean_L = mean(L_1, na.rm = TRUE), .groups = "drop")
    
    old_par <- par(mfrow = c(2, 1), mar = c(3, 4, 2, 1))
    barplot(setNames(summary_df$n, month.abb),
            main = "Observations per month", col = "steelblue", las = 1)
    plot(1:12, summary_df$mean_L, type = "b", pch = 19,
         xaxt = "n", xlab = "", ylab = "Mean L* (dominant color)",
         main = "Mean lightness by month")
    axis(1, at = 1:12, labels = month.abb)
    par(old_par)
  })
  
  # --- Raw data table ---
  output$data_table <- renderDT({
    md <- map_data()
    if (nrow(md) == 0) return(datatable(md))
    md %>%
      select(gbif_id, observation_date, month, n_photos,
             class_coverage, proportion_1, hex_1, proportion_2, hex_2,
             latitude, longitude) %>%
      datatable(options = list(pageLength = 15))
  })
}

shinyApp(ui, server)