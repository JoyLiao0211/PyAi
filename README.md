# Final project
train datas are too big they exceed github's file size limit
## Tasks
### 1-a
Done with `make_gray_level_histogram.py`, results in `image/gray_level_histogram/`.
### 1-b
Done with `gen_image.py`, results in `image/image_origin`.
### 1-c
Also done with `gen_image.py`, results in `image/image_cross`.

### 2-a

Done with `find_rows_with_missing_data.py`, indice of images that contain invalid or missing data are listed below.
Also I generated the images with `create_preview.py` with invalid data marked as red and the images are at `image/image_with_missing/`.

```
fashion test
10 : [1877, 2516, 2707, 2910, 3760, 3918, 6027, 8347, 8582, 9368]
mnist test
10 : [326, 1801, 2389, 3300, 3392, 4122, 5403, 7710, 7928, 9490]
```

### 2-b

#### naive approach: drop the rows & mean and median of rows
Done with `fill_missing_data_naive.py`, results in `{fashion/mnist}_test_col_{mean/median}.csv`, `{fashion/mnist}_test_drop.csv` and at `image/column_{mean/median}`.

#### Image processing approach: mean and median of N4 and N8 of the pixel
N4: 4 neighbors of a pixel
N8: 8 neighbors of a pixel
Done with `fill_missing_data_N4_N8.py`, results in `{fashion/mnist}_test_{N4/N8}_{mean/median}.csv` and at `image/{N4/N8}_{mean/median}/`
