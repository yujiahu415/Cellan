from readlif.reader import LifFile
from tifffile import imread,imwrite
import os
import numpy as np
import cv2
from skimage import exposure



def extract_images(path_to_file,out_folder,fov_dim,imagewidth=None,black_background=True):

	if os.path.splitext(os.path.basename(path_to_file))[1] in ['.svs','.SVS']:

		image=imread(path_to_file)

		width=image.shape[1]
		height=image.shape[0]
		num_w=int(width/fov_dim)
		num_h=int(height/fov_dim)
		if black_background:
			background=np.zeros((fov_dim,fov_dim,3),dtype='uint8')
		else:
			background=np.uint8(np.ones((fov_dim,fov_dim,3))*255)

		for h in range(num_h):

			for w in range(num_w):

				fov=np.uint8(exposure.rescale_intensity(image[int(h*fov_dim):min(int((h+1)*fov_dim),height)+1,int(w*fov_dim):min(int((w+1)*fov_dim),width)+1],out_range=(0,255)))
				if fov.shape[0]<fov_dim or fov.shape[1]<fov_dim:
					background[fov]=fov
					fov=background
				if imagewidth is not None:
					fov=cv2.resize(fov,(imagewidth,imagewidth),interpolation=cv2.INTER_AREA)
				cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'.png'),fov)

	elif os.path.splitext(os.path.basename(path_to_file))[1] in ['.tif','.TIF','.tiff','.TIFF']:

		image=imread(path_to_file)

		fov_width=int(image.shape[1]/fov_dim)
		fov_height=int(image.shape[0]/fov_dim)

		if len(list(image.shape))<3:
			c_list=None
		else:
			c_list=[0,1,2]

		for w in range(fov_dim):

			for h in range(fov_dim):

				if c_list is None:
					fov=np.uint8(exposure.rescale_intensity(image[h*fov_height:(h+1)*fov_height,w*fov_width:(w+1)*fov_width],out_range=(0,255)))
					if imagewidth is not None:
						fov=cv2.resize(fov,(imagewidth,int(fov.shape[0]*imagewidth/fov.shape[1])),interpolation=cv2.INTER_AREA)
					cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'.png'),fov)
				else:
					for c in c_list:
						fov=np.uint8(exposure.rescale_intensity(image[h*fov_height:(h+1)*fov_height,w*fov_width:(w+1)*fov_width,c],out_range=(0,255)))
						if imagewidth is not None:
							fov=cv2.resize(fov,(imagewidth,int(fov.shape[0]*imagewidth/fov.shape[1])),interpolation=cv2.INTER_AREA)
						cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'_c'+str(c)+'.png'),fov)

	elif os.path.splitext(os.path.basename(path_to_file))[1] in ['.qptiff','.QPTIFF']:

		image=imread(path_to_file)

		c_list=[i for i in range(image.shape[0])]
		fov_width=int(image.shape[2]/fov_dim)
		fov_height=int(image.shape[1]/fov_dim)

		for c in c_list:

			for w in range(fov_dim):

				for h in range(fov_dim):

					fov=np.uint8(exposure.rescale_intensity(image[c,h*fov_height:(h+1)*fov_height,w*fov_width:(w+1)*fov_width],out_range=(0,255)))
					if imagewidth is not None:
						fov=cv2.resize(fov,(imagewidth,int(fov.shape[0]*imagewidth/fov.shape[1])),interpolation=cv2.INTER_AREA)
					imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'_c'+str(c)+'.png'),fov)

	else:

		lifdata=LifFile(path_to_file)
		file=[i for i in lifdata.get_iter_image()][0]

		c_list=[i for i in file.get_iter_c(t=0,z=0)]

		image=np.array(file.get_frame(z=0,t=0,c=0))
		fov_width=int(image.shape[1]/fov_dim)
		fov_height=int(image.shape[0]/fov_dim)

		for w in range(fov_dim):

			for h in range(fov_dim):

				for c in range(len(c_list)):

					image=np.array(file.get_frame(z=0,t=0,c=c))
					fov=np.uint8(exposure.rescale_intensity(image[h*fov_height:(h+1)*fov_height,w*fov_width:(w+1)*fov_width],out_range=(0,255)))
					if imagewidth is not None:
						fov=cv2.resize(fov,(imagewidth,int(fov.shape[0]*imagewidth/fov.shape[1])),interpolation=cv2.INTER_AREA)
					cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'_c'+str(c)+'.png'),fov)


