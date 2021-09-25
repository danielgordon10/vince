# Video Noise Contrastive Estimation (VINCE)
This is a repository containing code used to implement the models in the paper  
[Watching the World Go By: Representation Learning from Unlabeled Videos](https://danielgordon10.github.io/pdfs/vince.pdf) (https://arxiv.org/abs/2003.07990).

<img src="https://danielgordon10.github.io/images/projects/vince.jpg" height="500"/>

## Environment Setup
We recommend using `Anaconda` to manage your environment setup and run our code. 
The following commands will create an environment similar to ours with minimal requirements.
### Conda
```bash
conda create -n video-env python=3.6.8
conda deactivate
conda env update -n video-env -f env.yml
conda activate video-env
pip install git+https://github.com/danielgordon10/dg_util.git -U
```

### Virtualenv
If you instead prefer `virtualenv` or similar, we have also provided a [requirements.txt](requirements.txt).
```bash
virtualenv --python=python3.6 video-env
source video-env/bin/activate
pip install -r requirements.txt
```

## Downlaod Random Related Video Views (R2V2)
The dataset released is larger than the one presented in the paper. Each image denotes its source video and frame number in its file name. For example `aa3jheRwEYo_000725.jpg` corresponds to video `https://www.youtube.com/watch?v=aa3jheRwEYo` frame `725`.

To download one large file containing all the train and val images, download https://drive.google.com/open?id=1JGwb6ai1NugeW7KJDDE7G_bZ54D69teF

Alternatively:
1. To download via the command line,  run `python download_scripts/download_r2v2.py`.
1. This may fail to download some of the files due to Google drive rate limiting, but you may still be able to download them via the web browser. You will have to manually download the links in [r2v2_drive_urls.txt](datasets/info_files/r2v2_drive_urls.txt).

### Notes
|       | Size (GB) | Number of Files | Number of Images | Number of Folders | Number of Source Videos |
|-------|:---------:|:---------------:|:----------------:|:-----------------:|:-----------------------:|
| Train |       110 |       2,788,424 |        2,784,328 |              4096 |                 696,082 |
| Val   |       8.8 |         226,620 |          222,524 |              4096 |                  55,631 |
 - Some folders have many more images than others. This is expected.
 - The video and frame ids are also provided in [datasets/info_files/r2v2_ids_train.txt](datasets/info_files/r2v2_ids_train.txt) and [datasets/info_files/r2v2_ids_val.txt](datasets/info_files/r2v2_ids_val.txt)  
 
### Downloading your own set of YouTube videos
If you would like to download a different set of YouTube videos, you may still find our code helpful.
Here is a basic workflow for downloading many YouTube videos.

1. [Create cookies.txt](#create-cookies.txt)
1. Create a list of many YouTube URLs to download. 
    1. One option would be to use [youtube_scrape/search_youtube_for_urls.py](youtube_scrape/search_youtube_for_urls.py)
    1. Another would be YouTube-8m URLs (https://github.com/danielgordon10/youtube8m-data)
1. Run `python run_cache_video_dataset.py --title cache --description caching --num-workers 100` after appropriately formatting the files.  
    - Note - You can often use more workers than your CPU has threads because YouTube downloading tends to be the bottleneck.
1. [youtube_scrape/download_kinetics.py](youtube_scrape/download_kinetics.py) is a convenient file for downloading Kinetics videos.


### Create cookies.txt
1. Install [this extension](https://chrome.google.com/webstore/detail/cookiestxt/njabckikapfpffapmjgojcnbfjonfjfg).
1. Go to any youtube video: https://www.youtube.com/watch?v=AKQE9RyOIMY
1. Click the cookie icon and save the data into `youtube_scrape/cookies.txt` or adjust the `COOKIE_PATH` variable in [constants.py](constants.py)


## Training
### Train VINCE
1. [Download R2V2 training data](#download-r2v2) or [create your own dataset](#downloading-your-own-set-of-youtube-videos) to train on.
1. Read over the arguments list in [arg_parser.py](arg_parser.py).
1. Train the model. We have provided an example [train script](vince/train_vince.sh) as well as a [debug script](vince/train_vince_debug.sh) to check everything is working. Edit the paths in the file to point to your data/output locations.

### Train baselines
1. The official MoCo baseline is available at https://github.com/facebookresearch/moco, but for our work, we wrote our own version.
1. We have provided an example [train script](vince/train_moco_baseline.sh) to train this model.
1. We additionally include MoCoV2 baseline scripts for ResNet50 at [vince/train_moco_v2.sh](vince/train_moco_v2.sh).
1. We additionally include the Jigsaw method from [PIRL](https://arxiv.org/abs/1912.01991) and an accompanying script [vince/train_vince_jigsaw.sh](vince/train_vince_jigsaw.sh). Pretrained weights and results are currently not provided.

### Train End Task
1. We include various end tasks and an interface for easily adding more. Training scripts for each task are available at:
    1. [end_tasks/train_imagenet.sh](end_tasks/train_imagenet.sh)
    1. [end_tasks/train_sun_scene.sh](end_tasks/train_sun_scene.sh)
    1. [end_tasks/train_kinetics_400.sh](end_tasks/train_kinetics_400.sh)
    1. [end_tasks/train_tracking.sh](end_tasks/train_tracking.sh)
1. New end tasks can be added by creating a new solver which inherits from [EndTaskBaseSolver](solvers/end_task_base_solver.py) and an accompanying dataset which inherits from [BaseDataset](datasets/base_dataset.py).


## Evaluation
1. While training each end task, evaluation is done after every epoch on a val set.
1. If more evaluation is needed, it can be added by implementing `run_eval` for that solver. For an example, see [solvers/end_task_tracking_solver.py](solvers/end_task_tracking_solver.py) and [end_tasks/eval_tracking.sh](end_tasks/eval_tracking.sh).

### Download Pretrained Weights
Pretrained weights are available for VINCE as well as all baselines mentioned in the paper. 
We provide the pretrained weights for the backbone only, not for any end task.

#### ResNet18
To download the weights, from the root directory, run `sh download_scripts/download_pretrained_weights_resnet18.sh`
Alternatively, download them directly from https://drive.google.com/uc?id=1QYuUgdNkhOIdy3hle79uWaJER4Z7SIlD

#### ResNet50
These models were trained using the hyperparameters in https://arxiv.org/abs/2003.04297 except for batch size which was 896 (starting loss was scaled proportionally to 0.105).
To download the weights, from the root directory, run `sh download_scripts/download_pretrained_weights_resnet50.sh`
Alternatively, download them directly from https://drive.google.com/uc?id=1c6wUtYZuCI_NAEhwtzB3j5F8yH8TNkZ3

#### Benchmark Results
The results you achieve should somewhat match the table below, though different learning schedules and other factors may slightly change performance.

| Method Name (In Paper) | Dir Name                          | Backbone | ImageNet | Sun Scenes | Kinetics 400 | OTB 2015 Precision | OTB 2015 Success |
|------------------------|-----------------------------------|:--------:|:--------:|:----------:|:------------:|--------------------|:----------------:|
| Sup-IN                 | N/A                               | ResNet18 |    0.696 |      0.491 |        0.207 |              0.557 |            0.396 |
| MoCo-IN                | moco-in                           | ResNet18 |    0.447 |      0.487 |        0.336 |              0.583 |            0.429 |
| MoCo-G                 | moco-g                            | ResNet18 |    0.393 |      0.444 |        0.313 |              0.511 |            0.413 |
| MoCo-R2V2              | moco-r2v2                         | ResNet18 |    0.358 |      0.450 |        0.318 |              0.555 |            0.403 |
| VINCE                  | vince-r2v2-multi-frame-multi-pair | ResNet18 |    0.400 |      0.495 |        0.362 |              0.629 |            0.465 |
| Sup-IN                 | N/A                               | ResNet50 |    0.762 |      0.593 |        0.305 |              0.458 |            0.320 |
| MoCo-V2-IN             | moco-v2-in                        | ResNet50 |    0.652 |      0.608 |        0.459 |              0.300 |            0.260 |
| MoCo-R2V2              | moco-v2-r2v2                      | ResNet50 |    0.536 |      0.581 |        0.456 |              0.386 |            0.299 |
| VINCE                  | vince-r2v2-multi-frame-multi-pair | ResNet50 |    0.544 |      0.611 |        0.491 |              0.402 |            0.300 |

## Citation
```
@misc{gordon2020watching,
    title={Watching the World Go By: Representation Learning from Unlabeled Videos},
    author={Gordon, Daniel and Ehsani, Kiana and Fox, Dieter and Farhadi, Ali},
    year={2020},
    eprint={2003.07990},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
