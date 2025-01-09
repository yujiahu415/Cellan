from readlif.reader import LifFile
from tifffile import imread
import os
import numpy as np
import cv2
from skimage import exposure



def extract_images(path_to_file,out_folder,fov_div,imagewidth=None):

	if os.path.splitext(os.path.basename(path_to_file))[1] in ['.svs','.SVS']:

		image=imread(path_to_file)

		fov_width=int(image.shape[1]/fov_div)
		fov_height=int(image.shape[0]/fov_div)

		for w in range(fov_div):

			for h in range(fov_div):

				fov=np.uint8(exposure.rescale_intensity(image[h*fov_height:(h+1)*fov_height,w*fov_width:(w+1)*fov_width],out_range=(0,255)))
				if imagewidth is not None:
					fov=cv2.resize(fov,(imagewidth,int(fov.shape[0]*imagewidth/fov.shape[1])),interpolation=cv2.INTER_AREA)
				cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'.png'),fov)

	elif os.path.splitext(os.path.basename(path_to_file))[1] in ['.tif','.TIF','.tiff','.TIFF']:

		image=imread(path_to_file)

		fov_width=int(image.shape[1]/fov_div)
		fov_height=int(image.shape[0]/fov_div)

		if len(list(image.shape))<3:
			c_list=None
		else:
			c_list=[0,1,2]

		for w in range(fov_div):

			for h in range(fov_div):

				if c_list is None:
					fov=np.uint8(exposure.rescale_intensity(image[h*fov_height:(h+1)*fov_height,w*fov_width:(w+1)*fov_width],out_range=(0,255)))
					if imagewidth is not None:
						fov=cv2.resize(fov,(imagewidth,int(fov.shape[0]*imagewidth/fov.shape[1])),interpolation=cv2.INTER_AREA)
					cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'.png'),fov)
				else:
					for c in range(len(c_list)):
						fov=np.uint8(exposure.rescale_intensity(image[h*fov_height:(h+1)*fov_height,w*fov_width:(w+1)*fov_width,c],out_range=(0,255)))
						if imagewidth is not None:
							fov=cv2.resize(fov,(imagewidth,int(fov.shape[0]*imagewidth/fov.shape[1])),interpolation=cv2.INTER_AREA)
						cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'_c'+str(c)+'.png'),fov)

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
					if imagewidth is not None:
						fov=cv2.resize(fov,(imagewidth,int(fov.shape[0]*imagewidth/fov.shape[1])),interpolation=cv2.INTER_AREA)
					cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'_c'+str(c)+'.png'),fov)


