```python
%%shell

# Download TorchVision repo to use some files from
# references/detection
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0

cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
```

由于本地运行的jupyter使用不了上面的代码，所以直接将文件下载到平行的文件夹方便import。

## 说明

- `%%shell`

  以Linux控制台的形式运行代码。

- `cp` copy file 从一个地址到一个地址

- `../`与当前文件夹平行。