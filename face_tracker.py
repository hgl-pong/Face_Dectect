import cv2
import sys
import os
import io
import matplotlib.pyplot as plt


data_types= ['OTB','VOT']
data_type=data_types[0]

vedio_name="Dudek"

use_mode=0    #使用模式，当为0时为使用groundtruth人工标注数据进行初始化（可以绘制曲线），当为1时为自主框选追踪物体

#——————openCv追踪方法3.4
tracker_types = ['KCF', 'TLD']  #'KCF', 'TLD'
tracker_type = tracker_types[0]
global tracker
# 创建跟踪器
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()



# 创建窗口
# cv2.namedWindow("Face_Tracking")

## 读入视频
#video = cv2.VideoCapture("./data/1.mp4")
## 读入第一帧
#ok, frame = video.read()

#————————读取图片序列——————————————
img_root="./data/"+data_type+"/"+vedio_name+"/img/"
frame=[]
im_names=os.listdir(img_root)
im_names.sort(key=lambda x:int(x[:-4]))
for im_name in im_names:  
    frame.append(cv2.imread(img_root+im_name))

if not frame:
    print('Cannot read video file')
    sys.exit()

#————————读取groundtruth_rect的box数据————————————
if use_mode == 0:
    if data_type == 'OTB':
        with io.open('./data/'+data_type+'/'+vedio_name+'/groundtruth_rect.txt') as f:
            lines = f.readlines()
    if data_type == 'VOT':
        with io.open('./data/'+data_type+'/'+vedio_name+'/groundtruth.txt') as f:
            lines = f.readlines()
    boxes=[]
    for line in lines:
        if data_type == 'OTB':  #数据格式为(x,y,width,height)
            #x,y,width,height=line.rstrip().split('\t')
            x,y,width,height=line.rstrip().split(',')
            box=(float(x),float(y),float(width),float(height))
            boxes.append(box)
        if data_type == 'VOT':  #数据格式为矩形的4个点坐标(x1,y1,x2,y2,x3,y3,x4,y4)
            x1,y1,x2,y2,x3,y3,x4,y4=line.rstrip().split(',')
            xmin=min(float(x1),float(x2),float(x3),float(x4))
            ymin=min(float(y1),float(y2),float(y3),float(y4))
            xmax=max(float(x1),float(x2),float(x3),float(x4))
            ymax=max(float(y1),float(y2),float(y3),float(y4))
            box=(xmin,ymin,xmax-xmin,ymax-ymin)
            print(box)
            boxes.append(box)
    # 用第一帧初始化
    tracker.init(frame[0],boxes[0])
if use_mode == 1:
    # 自定义一个bounding box
    #bbox = (287, 23, 86, 320)  #根据坐标定义
    box1 = cv2.selectROI("Face_Tracking", frame[0])  #根据鼠标框选区域定义
    # 用第一帧初始化
    tracker.init(frame[0],box1)


# 求两个矩形的重叠率（交集/并集）
# box=(x,y,width,height)
def bbOverlap(box1,box2):
    if box1[0] > box2[0]+box2[2]: return 0.0
    if box1[1] > box2[1]+box2[3]: return 0.0
    if box1[0]+box1[2] < box2[0]: return 0.0
    if box1[1]+box1[3] < box2[1]: return 0.0
    colInt =  min(box1[0]+box1[2],box2[0]+box2[2]) - max(box1[0], box2[0])
    rowInt =  min(box1[1]+box1[3],box2[1]+box2[3]) - max(box1[1],box2[1])
    intersection = colInt * rowInt
    area1 = box1[2]*box1[3]
    area2 = box2[2]*box2[3]
    return intersection / (area1 + area2 - intersection)

# 求两个矩形中心点之间的距离
# box=(x,y,width,height),xy为矩形左上角坐标
def getDistance(box1,box2):
    x=abs(box1[0]+box1[2]/2-(box2[0]+box2[2]/2))
    y=abs(box1[1]+box1[3]/2-(box2[1]+box2[3]/2))
    return ((x ** 2) + (y ** 2)) ** 0.5

# ———————————定义绘制曲线图方法——————————————
#总绘制方法
def drawfig(x,y,fig_title,y_label):
     fig  = plt.figure()
     #plt.xlim(0,1)   #设置坐标轴范围
     plt.ylim(0,1)
     fig.suptitle(fig_title, fontsize = 14, fontweight='bold')  #设置图表标题
     ax = fig.add_subplot(1,1,1)  #增加子图，1个子图，1×1类型
     ax.set_xlabel("threshold")   #设置x,y轴名称
     ax.set_ylabel(y_label)
     ax.plot(x,y)
     plt.show()
#绘制Success Rate曲线
def drawSuccessRate(overlap_score):
    threshold=0.0  #设置初始阈值（依次递增）
    SuccessRate=[] # y轴
    overlap_threshold=[]  # x轴
    while threshold<=1:
        success_count=0  #记录在当前阈值下，标记为成功的图像数
        for score in overlap_score:
            if score > threshold: success_count+=1
        SuccessRate.append(success_count/len(overlap_score))
        overlap_threshold.append(threshold)
        threshold+=0.02
    # 将计算完的x,y写入本地文件中
    for i in range(len(overlap_threshold)):
        f = open('./out/'+'SR-'+vedio_name+'-'+tracker_type+'.txt','a',encoding='utf-8')
        f.write(str(overlap_threshold[i])+'\t'+str(SuccessRate[i])+'\n')
        f.close()
    drawfig(overlap_threshold,SuccessRate,"Success plots of OPE","Success rate")
#绘制Precision Rate曲线
def drawPrecisionRate(distances):
    threshold=0.0  #设置初始阈值（依次递增）
    Precision=[] # y轴
    Location_error_threshold=[]  # x轴
    while threshold<=50:
        Precision_count=0  #记录在当前阈值下，标记为成功的图像数
        for distance in distances:
            if distance < threshold: Precision_count+=1
        Precision.append(Precision_count/len(distances))
        Location_error_threshold.append(threshold)
        threshold+=1
     # 将计算完的x,y写入本地文件中
    for i in range(len(Location_error_threshold)):
        f = open('./out/'+'PR-'+vedio_name+'-'+tracker_type+'.txt','a',encoding='utf-8')
        f.write(str(Location_error_threshold[i])+'\t'+str(Precision[i])+'\n')
        f.close()
    drawfig(Location_error_threshold,Precision,"Precision plots of OPE","Precision Rate")


overlap_score=[]  #重合率得分数组
distances=[]   # bounding box中心点之间的距离数组
FPSs=[]  #FPS数组

index=1   #标识当前帧

#——————开始逐帧检测——————————————
while True:
    if index==len(frame):  #到最后一帧
       break
    curframe = frame[index]

    # Start timer
    timer = cv2.getTickCount()
    # Update tracker
    ok, track_boxes = tracker.update(curframe)

    # Cakculate FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    FPSs.append(fps)

    if use_mode == 0:
        # 计算重合率得分 overlap_score 数据
        overlap_score.append(bbOverlap(track_boxes,boxes[index]))
        # 计算中心距离数据
        distances.append(getDistance(track_boxes,boxes[index]))

    #——————单目标跟踪————————
    # if ok:
    #     (x, y, w, h) = [int(v) for v in track_boxes]
    #     cv2.rectangle(curframe, (x, y), (x + w, y + h),(255, 0, 0), 1)

    ##——————多目标跟踪————————
    for box in boxes:
       # Draw bonding box
       if ok:
           p1 = (int(box[0]), int(box[1]))
           p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
           cv2.rectangle(curframe, p1, p2, (255, 0, 0), 2, 1)
       else:
           cv2.putText(curframe, "Tracking failed detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),2)
        
    # 展示tracker类型
    cv2.putText(curframe, tracker_type+"Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # 展示FPS
    cv2.putText(curframe, "FPS:"+str(fps), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # Result
    cv2.imshow("Face_Tracking",curframe)
    # 等待渲染完成继续下一帧
    k = cv2.waitKey(10) & 0xff
    if k ==27 : break

    index+=1

if use_mode == 0:
    #执行绘制曲线函数
    drawPrecisionRate(distances)
    drawSuccessRate(overlap_score)

#导出fps
for i in range(len(FPSs)):
    f = open('./out/'+'FPS-'+vedio_name+'-'+tracker_type+'.txt','a',encoding='utf-8')
    f.write(str(i)+'\t'+str(FPSs[i])+'\n')
    f.close()