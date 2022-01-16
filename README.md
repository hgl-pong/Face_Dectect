# 环境依赖
 opencv3.4版本\
 opengl\
 glfw
# 部署步骤
1. 安装opencv
    ```
    pip install opencv-contrib-python==3.4.8.29 -i https://pypi.tuna.tsinghua.edu.cn/simple    
    ```

2. 安装opengl
    ```
    pip install pyopengl
    pip install pyopengl-accelerate
    ```

3. 安装glfw
    ```
    pip install glfw
    ```

# 目录结构描述
```
│  face_detecter.py  //人脸识别模块
│  face_tracker.py   //人脸跟踪模块
│  HOG.py    //Hog特征提取计算
│  draw.py    //窗口绘制以及滤镜添加
│  
├─.github
│  └─workflows
│          Post.yaml   //git actions
│          
├─data
│  ├─OTB
│  │  └─Dudek      //OTB数据集（视频帧）
│  │      │  groundtruth_rect.txt    //人工标记
│  │      └─img
│  └─VOT
└─out   //算法输出文件
```

