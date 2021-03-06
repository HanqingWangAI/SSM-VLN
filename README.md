# SSM-VLN

Code and Data for our CVPR 2021 paper "[Structured Scene Memory for Vision-Language Navigation](https://arxiv.org/abs/2103.03454)".



## Environment Installation
Download Room-to-Room navigation data:
```
bash ./tasks/R2R/data/download.sh
```

Download image features for environments:
```
mkdir img_features
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip -P img_features/
cd img_features
unzip ResNet-152-imagenet.zip
```

Python requirements: Need python3.6.
```
conda create -n ssm python=3.6
conda activate ssm
pip install -r python_requirements.txt
```

Install Matterport3D simulators:
```
git submodule update --init --recursive 
sudo apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make -j8
```

## Usage

### Agent Training
``` bash
cd ssm
python train.py
```


### Agent Evaluation
Run the following scripts to evaluate the checkpoints.
``` bash
cd ssm
python eval_agent.py
```

The trained model for R2R task is available in [GoogleDrive](https://drive.google.com/file/d/15mINW_HOxweO-OX2W-LZN_5YoZXpzmNV/view?usp=sharing). Please download the checkpoint file under `snap/SSM/state_dict` path and run the following script to evaluate the model.
``` bash
cd ssm
python model_eval.py
```



## Citation
Please cite this paper in your publications if it helps your research:
```
@inproceedings{wang2021structured,
      title={Structured Scene Memory for Vision-Language Navigation}, 
      author={Hanqing Wang and Wenguan Wang and Wei Liang and Caiming Xiong and Jianbing Shen},
      booktitle=CVPR,
      year={2021}
}
```



<!-- 
## TODO's
1. [x] Release the checkpoint.
2. [x] Update the installation requirements.
3. [x] Add evaluation scripts. -->


## Contact Information
- hanqingwang[at]bit[dot]edu[dot]cn, Hanqing Wang
- wenguanwang[dot]ai[at]gmail[dot]com, Wenguan Wang