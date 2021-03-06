# MATE
The paper `Deep Behavior Tracing with Multi-level Temporality Preserved Embedding` is accepted by the 29th ACM International Conference on Information and Knowledge Management ([CIKM '20](https://www.cikm2020.org/)).

## Authors
Runze Wu (NetEase Fuxi AI Lab), Hao Deng (NetEase Fuxi AI Lab), Jianrong Tao (NetEase Fuxi AI Lab), Changjie Fan (NetEase Fuxi AI Lab), Qi Liu (University of Science and Techonology of China) and Liang Chen (Sun Yat-sen University).


## Abstract
Behavior tracing or predicting is a key component in various application scenarios like online user modeling and ubiquitous computing, which significantly benefits the system design (e.g., resource pre-caching) and improves the user experience (e.g., personalized recommendation). 
		Traditional behavior tracing methods like Markovian and sequential models take recent behaviors as input and infer the next move by using the most real-time information.
		However, these existing methods rarely comprehensively model the low-level \textit{temporal irregularity} in the recent behavior sequence, i.e., the unevenly distributed time intervals between consecutive behaviors, and the high-level \textit{periodicity} in the long-term activity cycle, i.e., the periodic behavior patterns of each user.
		
In this paper, we propose an intuitive and effective embedding method called \textbf{M}ulti-level \textbf{A}ligned \textbf{T}emporal \textbf{E}mbedding (MATE), which can tackle the temporal irregularity of recent behavior sequence and then align with the long-term periodicity in the activity cycle.
		Specifically, we combine time encoding and decoupled attention mechanism to build a temporal self-attentive sequential decoder to address the behavior-level temporal irregularity.
		To embed the activity cycle from the raw behavior sequence, we employ a novel temporal dense interpolation followed by a self-attentive sequential encoder.
		Then we first propose the periodic activity alignment to capture the long-term activity-level periodicity and construct the activity-behavior alignment to combine the activity-level with behavior-level representation to make the final prediction.
		We experimentally prove the effectiveness of the proposed model on a game player behavior sequence dataset and a real-world App usage trace dataset.
		Further, we deploy the proposed behavior tracing model into a game scene preloading service which can effectively reduce the waiting time of scene transfer by preloading the predicted game scene for each user.

## Background
![](https://noterminus.gitee.io/image_bed/images/MATE-CIKM20-MapPreloadingExample.png)
<center>Game Scene Preloading in MMORPGs</center>

## Model Architecture
![](https://noterminus.gitee.io/image_bed/images/MATE-CIKM20-MATE_MODEL.png)

## Performance Comparison
![](https://noterminus.gitee.io/image_bed/images/MATE-CIKM20-Performance.png)

## Online Service
![](https://noterminus.gitee.io/image_bed/images/MATE-CIKM20-OnlineService.png)

## Reference
Please use the following bib
```
@inproceedings{wu2020mate,
	title={Deep Behavior Tracing with Multi-level Temporality Preserved Embedding},
	author={Wu, Runze and Deng, Hao and Tao, Jianrong and Fan, Changjie and Liu, Qi and Chen, Liang},
	booktitle={Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20)},
	year={2020}
}
```

