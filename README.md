
# GCDance: Genre-Controlled 3D Full Body Dance Generation Driven By Music

<div align="center">
<a href='https://xinranliu7715.github.io/gcdance/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
<a href='https://arxiv.org/abs/2502.18309'><img src='https://img.shields.io/badge/ArXiv-2502.18309-red'></a> 

![GCDance cover](images/top-1.jpg)
</div>

Music driven dance generation is challenging because a model must respect genre conventions, preserve physical realism, and synchronize movements with beats and rhythm at high precision. Despite recent progress in music conditioned generation, many methods still struggle to express distinctive genre specific style. We present GCDance, a diffusion based framework for genre specific 3D full body dance generation conditioned on music and descriptive text. The approach introduces a text based control mechanism that converts prompts, including explicit genre labels and free form descriptions, into genre specific control signals, enabling accurate and controllable synthesis of genre consistent motion. To strengthen cross modal alignment between audio and text, we incorporate representations from a music foundation model, which leads to coherent and semantically aligned dance. We further propose a multi task optimization strategy that balances the extraction of text genre information with motion quality by jointly optimizing physical realism, spatial accuracy, and text classification. Extensive experiments on FineDance and AIST++ demonstrate that GCDance outperforms state of the art methods. The source code and demonstration videos are available online.





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
