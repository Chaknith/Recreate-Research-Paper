## Model

The model classifies one pixel at a time using a long spectral vector of the hyperspectral image.

```
Pixel at (r, c)
[0.12, 0.15, 0.18, ..., 0.43]
 └──────── 200 spectral bands ────────┘
```

## Data

- 1.3 meters resolution

[Link to all the datasets](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

### Pavia Centre scene

- 102 spectrals

|         # | class_name               |     samples |
| --------: | ------------------------ | ----------: |
|         1 | Water                    |      65,971 |
|         2 | Trees                    |       7,598 |
|         3 | Asphalt                  |       3,090 |
|         4 | Self-Blocking Bricks     |       2,685 |
|         5 | Bitumen                  |       6,584 |
|         6 | Tiles                    |       9,248 |
|         7 | Shadows                  |       7,287 |
|         8 | Meadows                  |      42,826 |
|         9 | Bare Soil                |       2,863 |
| **Total** | **All labelled classes** | **148,152** |

> Note: The number of samples differs from the number shown on the website. I have emailed the dataset coordinator regarding this discrepancy.

![PC Overlay](results/PC/PC_Overlay.jpg)

Overlay picture of the Pavia Centre scene.

### Pavia University scene

- 103 spectrals

|         # | class_name               |    samples |
| --------: | ------------------------ | ---------: |
|         1 | Asphalt                  |      6,631 |
|         2 | Meadows                  |     18,649 |
|         3 | Gravel                   |      2,099 |
|         4 | Trees                    |      3,064 |
|         5 | Painted metal sheets     |      1,345 |
|         6 | Bare Soil                |      5,029 |
|         7 | Bitumen                  |      1,330 |
|         8 | Self-Blocking Bricks     |      3,682 |
|         9 | Shadows                  |        947 |
| **Total** | **All labelled classes** | **42,776** |