
# GCDance: Genre-Controlled 3D Full Body Dance Generation Driven By Music

<div align="center">
<a href='https://xinranliu7715.github.io/gcdance/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
<a href='https://arxiv.org/abs/2502.18309'><img src='https://img.shields.io/badge/ArXiv-2502.18309-red'></a> 

![GCDance cover](images/top-1.jpg)
</div>
It is a challenging task to generate high-quality full-body dance
sequences from music, as this requires strict adherence to genre-
specific choreography while ensuring physically realistic and pre-
cisely synchronized dance sequences with the music’s beats and
rhythm. Although significant progress has been made in music-
conditioned dance generation, most existing methods struggle to
convey specific stylistic attributes in generated dance. To bridge this
gap, we propose a diffusion-based framework for genre-specific 3D
full-body dance generation, conditioned on both music and descrip-
tive text.





## Data preparation

In our experiments, we use FineDance dataset for both training and evaluation. Please visit [Google Drive](https://drive.google.com/file/d/1zQvWG9I0H4U3Zrm8d_QD_ehenZvqfQfS/view?usp=sharing) to download and download the required SMPL models from [here] (https://smpl-x.is.tue.mpg.de/) into './assets'.

### Data preparation
To process the motion data.
```python
python preprocess/pre_motion.py --motion_dir --store_dir
```
To process the music data.
```python 
python preprocess/pre_music.py --music_dir --store_dir
```


### Training
We provide two training modes:
#### Aligned Training
```python
accelerate launch train.py --wandb --mtl_method Aligned
```
#### Nash Training
```python
accelerate launch train.py --wandb --mtl_method Nash
```

### Generate

```python
python test.py --test_gen
```
### Evaluate

```python
python test.py --eval --type 0
```

### Visualization
```python
python vis.py --motion_save_dir 
```
