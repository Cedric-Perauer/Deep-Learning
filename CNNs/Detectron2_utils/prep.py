import os 
from PIL import Image

base = os.path.expanduser("~/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_03/data/")
new_base = os.path.expanduser("~/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_03/data_new/")
a = os.listdir(base)


for i in a: 
    f_name = base + i
    new_name = new_bae +  i.split(".")[0] + ".jpg"
    
    im1 = Image.open(f_name)
    im1.save(new_name)
