# Bay-CAT
<div align="center">

<h1> 湾大 Bay-CAT </h1>
<h2 class="papername"> <img src="./assets/logo.png" style="vertical-align: middle; height: 1em; padding: 0 0.2em;"> CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios </h2>
<div>
<div>
    <a href="https://scholar.google.com/citations?user=1joiJpUAAAAJ" target="_blank">Qilang Ye<sup>1</sup></a>,
    <a href="https://zitongyu.github.io/" target="_blank">Zitong Yu*<sup>1</sup></a>,
    <a href="https://rshaojimmy.github.io/" target="_blank">Rui Shao<sup>2</sup></a>,
    <a href="https://ieeexplore.ieee.org/author/37090029620" target="_blank">Xinyu Xie<sup>1</sup></a>,
    <a href="https://scholar.google.com/citations?user=kPxa2w0AAAAJ" target="_blank">Philip Torr<sup>3</sup></a>,
    <a href="https://scholar.google.com/citations?user=PDgp6OkAAAAJ" target="_blank">Xiaochun Cao<sup>4</sup></a>
</div>

<sup>1</sup> Great Bay University<br>
<sup>2</sup> Harbin Institute of Technology, Shenzhen<br>
<sup>3</sup> University of Oxford<br>
<sup>4</sup> Shenzhen Campus of Sun Yat-sen University<br>
* Corresponding author

[[Paper]]() [[Project Page]](https://github.com/rikeilong/Bay-CAT)

:fire: The codes and the collected instructions <b>AVinstruct</b> will be released. Stay tuned :beers: :+1: 

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fwww.slywiki.cn&count_bg=%2379C83D&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=Visitors&edge_flat=false)](https://hits.seeyoufarm.com)
</div>
<br>

## Introduction
We introduce the CAT, enhancing MLLM in three ways: 
1) we design a clue aggregator that aggregates question-related clues in dynamic audio-visual scenarios to enrich the detailed knowledge required for large language models. 
2) CAT is trained on a mixed multimodal dataset, allowing direct application in audio-visual scenarios. Notably, we collect an audio-visual joint instruction dataset named AVinstruct, to further enhance the capacity of CAT to model cross-semantic correlations.
3) we propose AI-assisted ambiguity-aware direct preference optimization, a strategy specialized in retraining the model to favor the non-ambiguity response and improve the ability to localize specific audio-visual objects.
  
<img src='assets/Introduction.jpg' width='90%'>

</div>

## Qualitative Results

![Qualitative Comparison](assets/app-visual-1.png)
![Qualitative Comparison](assets/app-visual-2.png)
![Qualitative Comparison](assets/app-visual-3.png)

## Updates
- [03/2024] [Project page](https://github.com/rikeilong/Bay-CAT) released.

## If you find this work useful for your research, please kindly cite our paper and star our repo.

## Citation
