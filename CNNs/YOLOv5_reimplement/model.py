import torch 
import torch.nn as nn
#short version of the code by Alladin Person 
#layers 
architecture_config = [
    (7, 64, 2, 3), #Filter size, filetr num , stride and padding 
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    #List : tuples, 4 is the num of repeats  
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], 
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNN_Block(nn.Module): 
    def __init__(self,in_channels,out_channels,**kwags): 
        super(CNN_Block,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwags) 
        self.bn = nn.BatchNorm2d(out_channels) 
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self,x): 
        x = self.conv(x)
        x = self.bn(x) 
        return  self.leaky_relu(x) 

class YOLOv1(nn.Module): 
    def __init__(self,in_channels=3,**kwags) :
        super(YOLOv1,self).__init__()
        self.architecture = architecture_config 
        self.in_channels = in_channels
        self.darknet = self.create_layers() 
        self.fc = self.create_fc(**kwags) #Yolov1 uses fcn layers 

    def forward(self,x): 
        x = self.darknet(x)
        flatten = torch.flatten(x,start_dim=1)
        return self.fc(flatten) 
    
    def create_layers(self): 
        layers = [] 
        in_channels = self.in_channels  
        for x in self.architecture: 
            if type(x) == tuple : 
                layers += [CNN_Block(
                                in_channels,
                                out_channels = x[1],
                                kernel_size = x[0],
                                stride = x[2],
                                padding = x[3])]
                in_channels = x[1]

            elif type(x) == str : 
                layers += [nn.MaxPool2d((2,2),stride=(2,2))] #always max pool 
            elif type(x) == list:
                num_repeats = x[2]
                for i in range(num_repeats):
                    for j in range(len(x)-1):
                        layers += [
                            CNN_Block(
                                in_channels,
                                x[j][1],
                                kernel_size=x[j][0],
                                stride=x[j][2],
                                padding=x[j][3],
                            )
                        ]
                        in_channels = x[j][1]
                    in_channels = x[1][1]
        return nn.Sequential(*layers) 



    def create_fc(self,split_size,num_boxes,num_classes): 
       S,B,C = split_size, num_boxes,num_classes
       return nn.Sequential(
               nn.Flatten(),
               nn.Linear(1024 * S * S,4096), 
               nn.Dropout(0.0),
               nn.Linear(4096,S * S * (C + B * 5)),   
               )


