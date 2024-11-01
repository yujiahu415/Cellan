from readlif.reader import LifFile
from tifffile import imread
import os
import numpy as np
import cv2
from skimage import exposure



def extract_images(path_to_file,out_folder):

	tif=False
	delta=5.0

	if tif:
		tifdata=imread(path_to_file)
		file=[i for i in tifdata]
	else:
		lifdata=LifFile(path_to_file)
		file=[i for i in lifdata.get_iter_image()][0]

	z_list=[i for i in files[0].get_iter_z(t=0,c=0)]
	c_list=[i for i in files[0].get_iter_c(t=0,z=0)]


	for c in range(len(c_list)):

		image=files[0].get_frame(z=0,t=0,c=c)
		image=np.array(image)

		fov_width=int(image.shape[0]/10)
		fov_height=int(image.shape[1]/10)

		for n in range(10):
			fov=image[n*fov_width:(n+1)*fov_width,n*fov_height:(n+1)*fov_height]
			fov=np.uint8(exposure.rescale_intensity(fov,out_range=(0,255)))

			fov=np.uint8(fov)
			cv2.imwrite(os.path.join(out_folder,str(c)+'_'+str(n)+'_new.png'),fov)