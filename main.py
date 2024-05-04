import tkinter as tk # 这个是用于GUI的包，里面包含了各式各样的空间/This is a package for GUI, which contains various spaces
import os
from PIL import Image,ImageTk
from models import LeNet , VGG9
from torchvision import transforms
import time
import torch
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_transform = transforms.Compose([
                transforms.Resize(32),  # 调整大小到指定尺寸/Resize to specified size
                transforms.ToTensor(),  # 转换为张量/Convert to tensor
                transforms.Normalize((0.5,),(0.5,)),  # 标准化/standardization
            ])

class ViewController(object):
    def __init__(self,model_type = 'lenet') -> None:
        # 加载模型/Load model
        self.model = LeNet() if model_type == 'lenet' else VGG9()
        state_dict = torch.load(f'results/{model_type}_100_best.pth',map_location='cpu')
        self.model.load_state_dict(state_dict) # 加载模型权重/Load model weights
        self.model.to(DEVICE)
    
        label_map = json.load(open('label_index.json'))
        self.idx2label = {v:k for k,v in label_map.items()}
        # 加载窗体信息/Load form information
        self.root = tk.Tk()
        self.root.title('Cargo sorting system') # 设置title，货物分拣系统/Set title, cargo sorting system
        self.root.geometry('1000x200')
        self.root.resizable(0,0) # 防止用户调整尺寸/Prevent users from resizing
        # 开始 & 停止按钮/start and stop button
        self.control_btn = tk.Button(self.root,text='Begin',command=self.fn_control_belt,bg='green',width=15)
        self.control_btn.place(x=800,y=10)
        # 绘制传送带/Draw conveyor belt
        self.belt_label = tk.Label(self.root,bg='white')
        self.belt_label.config(text='Conveyor belt',bg='#ECECEC')
        self.belt_label.place(x=850,y=175)
        self.belt_canvas = tk.Canvas(self.root, width=1000, height=20, bg="white")
        self.belt_canvas.place(x=0,y=150)
        num_segments = 100
        segment_width = int(self.belt_canvas.cget('width'))/ num_segments
        print(segment_width)
        self.belt_segments = []
        for i in range(num_segments):
            x0 = 0 + i * segment_width
            x1 = x0 + segment_width
            segment = self.belt_canvas.create_rectangle(
                x0, 0, x1, int(self.belt_canvas.cget('height')), 
                fill='gray', outline="black")
            self.belt_segments.append(segment)
        # 绘制检测空间/Draw detection space
        self.detect_canvas = tk.Canvas(self.root, width=120, height=40, bg="#ADD8E6")
        self.detect_canvas.place(x=100,y=20)
        hint_text = self.detect_canvas.create_text(
            int(self.detect_canvas.cget('width'))/2,int(self.detect_canvas.cget('height'))/2-10,
            text=' Detection Model ',fill='black',anchor=tk.CENTER)
        
        self.detech_text = self.detect_canvas.create_text(
            int(self.detect_canvas.cget('width'))/2,int(self.detect_canvas.cget('height'))/2+10,
            text=' Type ',fill='red',anchor=tk.CENTER)
    
        # 在传送带上绘制图片轮转/Make conveyor belt roll
        self.image_canvas = tk.Canvas(self.root, width=1000, height=80, bg="#ECECEC")
        self.image_canvas.place(x=0,y=70)
        ## 加载图片/loding photos

        image_dir = 'data/check_images/'
        image_files = os.listdir(image_dir)
        image_file_paths = [os.path.join(image_dir,file) for file in image_files]
        self.images_tk = []
        self.images_tensors = []
        for file in image_file_paths:
            print(file)
            image = Image.open(file)
            resized_image = image.resize((80,80))
            image_tensor = img_transform(image)
            image_tensor = image_tensor.to(DEVICE)
            photo_image = ImageTk.PhotoImage(resized_image)
            self.images_tk.append(photo_image)
            self.images_tensors.append(image_tensor)

        image_width , image_height = 80 , 80 # 正方形方块/Change the size of image 
        image_margin = 1000 / 10 - image_width # 10张图片，每个图片之间的间隔/the space between image
        self.image_segments = []
        for i in range(10):
            x0 = 0 + i * (image_width + image_margin)
            seg = self.image_canvas.create_image(x0,0,
                image=self.images_tk[i],anchor=tk.NW)
            self.image_segments.append(seg)

    def animate_conveyor_belt(self):
        # 模拟传送带滚动/Simulate conveyor belt rolling
        if self.control_btn.cget('text').lower() == 'stop':
            # 控制按钮显示停止，表示现在传送带正在转动/The control button shows Stop, indicating that the conveyor belt is now rotating
            for segment in self.belt_segments:  # 转动传送带里面的segments/Rotating segments inside the conveyor belt
                self.belt_canvas.move(segment, -10, 0)
            for image_seg in self.image_segments: # 转动图片/Rotate picture
                self.image_canvas.move(image_seg,-10,0)

            # 循环播放        
            for segment in self.belt_segments:
                x0, y0, x1, y1 = self.belt_canvas.coords(segment) # 坐标定位/Coordinate positioning
                if x1 <= 0:
                    self.belt_canvas.move(segment, self.belt_canvas.winfo_width(), 0) # 把最左边的图片转移到转移到右边\Move the leftmost image to the right
            for image_seg in self.image_segments:
                x0, y0 = self.image_canvas.coords(image_seg) # 坐标定位/Coordinate positioning
                x1 = x0+80
                if x1 <= 0:
                    self.image_canvas.move(image_seg, self.image_canvas.winfo_width(),0) # 把最左边的图片转移到右边/Move the leftmost image to the right
            # 检测模型/Detection
            for idx, image_seg in enumerate(self.image_segments):
                x0, y0 = self.image_canvas.coords(image_seg) # 坐标定位/Coordinate positioning
                #根据坐标来计算图片是否进入了检测装置\Calculate whether the picture has entered the detection device based on the coordinates
                detect_machine_x0 = self.detect_canvas.winfo_rootx()
                detect_machine_x1 = self.detect_canvas.winfo_rootx() + self.detect_canvas.winfo_width() / 2
                if x0 > detect_machine_x0 and x0 < detect_machine_x1:
                    tensor = self.images_tensors[idx]
                    tensor = tensor.unsqueeze(0)
                    output = self.model(tensor)
                    _,pred = torch.max(output,1)
                    pred = pred.view(-1)
                    pred = pred.cpu().detach().numpy().tolist()[0]
                    label_name = self.idx2label[int(pred)]
                    self.detect_canvas.itemconfig(self.detech_text,text=label_name)
                    print(f'Checking Result is: {label_name}')
            self.root.after(100, self.animate_conveyor_belt)

    ## 图片转动/image roll
    def fn_control_belt(self):
        # 函数：控制传送带\Function: Control conveyor belt
        if self.control_btn.cget('text').lower() == 'begin':
            print('Start the Conveyor Belt')
            self.control_btn.config(text='Stop')
            self.control_btn.config(bg='red')
            # 启动传送带动化\Start conveyorization
            self.animate_conveyor_belt()
        else:
            # 停止传送带\Stop
            print('Stop the Conveyor Belt')
            self.control_btn.config(text='Begin')
            self.control_btn.config(bg='green')

if __name__ == '__main__':
    vc = ViewController(model_type='lenet')
    vc.root.mainloop()
   
   