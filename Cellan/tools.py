from readlif.reader import LifFile
from tifffile import imread
import os
import numpy as np
import cv2
from skimage import exposure



def extract_images(path_to_file,out_folder,fov_div):

	if os.path.splitext(os.path.basename(path_to_file))[1] in ['.tif','.TIF','.tiff','.TIFF']:
		tifdata=imread(path_to_file)
		file=[i for i in tifdata]
	else:
		lifdata=LifFile(path_to_file)
		file=[i for i in lifdata.get_iter_image()][0]

	c_list=[i for i in file.get_iter_c(t=0,z=0)]

	image=np.array(file.get_frame(z=0,t=0,c=0))
	fov_width=int(image.shape[1]/fov_div)
	fov_height=int(image.shape[0]/fov_div)

	for w in range(fov_div):

		for h in range(fov_div):

			for c in range(len(c_list)):

				image=np.array(file.get_frame(z=0,t=0,c=c))
				fov=np.uint8(exposure.rescale_intensity(image[h*fov_height:(h+1)*fov_height,w*fov_width:(w+1)*fov_width],out_range=(0,255)))
				cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'_c'+str(c)+'.png'),fov)