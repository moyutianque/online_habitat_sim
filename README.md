# 


## data folder

```
data:
    mln3d -> /staging/leuven/stg_00095/zehao/mln3d
    scene_datasets -> /data/leuven/335/vsc33595/dataset/scene_datasets
```

## convert to video of output image sequence in ```out``` folder

```bash
ffmpeg -framerate 10 -i %d_rgb.jpg -c:v libx264 -vf fps=24 -pix_fmt yuv420p out.mp4
```