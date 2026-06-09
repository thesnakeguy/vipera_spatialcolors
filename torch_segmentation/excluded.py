"""Images excluded from ALL splits (train/val/test).

These 5 images carry NO snake annotation in `_annotations.coco.json`, yet every
one of them clearly DOES contain a *Vipera aspis* (verified 2026-06-06 by
inspecting the photos) -- including a melanistic individual eating a shrew
(437041559), a color morph the downstream study specifically cares about. So the
empty mask is a *labeling gap*, not a true "no snake here" negative.

Left in the data they do two kinds of damage:
  (a) poison training -- an all-zero mask teaches the model that snake pixels are
      background;
  (b) unfairly score iou_fg = 0.000 whenever one lands in val/test (empty GT vs a
      non-empty -- and correct! -- prediction), depressing the reported metric.

So we drop them from every split, at the split chokepoint (`data.split_image_ids`),
until they are re-annotated.

TODO(revisit): annotate these (rough polygons are fine) and delete the entry here
so the image rejoins train/val/test. The melanistic morph (813) in particular is
scientifically valuable and should be recovered first.

Keyed by COCO image id (stable within the current Roboflow export). File names are
kept alongside for human reference and to catch an id drift on re-export.
"""

# (coco image id, file_name, note) -- unlabeled snakes; see module docstring.
EXCLUDED = [
    (108, "431930628_jpg.rf.xBBuPosIWUaqsqwePLM1.jpg", "juvenile viper on bark"),
    (280, "432558057_jpg.rf.x1gUw63gZxAklwN50sbm.jpg", "viper coiled on sandy ground"),
    (727, "435845249_jpg.rf.LfJ1djo9Ww5GaZIrUaUj.jpg", "viper on grey rocks"),
    (752, "436986136_jpg.rf.O1v18Oz78rWetvFdyD3h.jpg", "pale snake on asphalt"),
    (813, "437041559_jpg.rf.D21WO6Me4rS0I1LzAa5G.jpg", "MELANISTIC viper eating a shrew"),
]

EXCLUDED_IDS = frozenset(i for i, _, _ in EXCLUDED)
EXCLUDED_FILES = frozenset(f for _, f, _ in EXCLUDED)


if __name__ == "__main__":
    print(f"{len(EXCLUDED)} excluded images (unlabeled snakes):")
    for i, f, note in EXCLUDED:
        print(f"  id={i:4d}  {f}  -- {note}")
