from readlif.reader import LifFile
from tifffile import imread,imwrite
import os
import numpy as np
import cv2
import pandas as pd
from skimage import exposure



def extract_images(path_to_file,out_folder,fov_dim,black_background=True):

	if os.path.splitext(os.path.basename(path_to_file))[1] in ['.svs','.SVS']:

		image=imread(path_to_file)

		width=image.shape[1]
		height=image.shape[0]
		num_w=int(width/fov_dim)
		if width%fov_dim!=0:
			num_w+=1
		num_h=int(height/fov_dim)
		if height%fov_dim!=0:
			num_h+=1

		for h in range(num_h):

			for w in range(num_w):

				fov=np.uint8(exposure.rescale_intensity(image[int(h*fov_dim):min(int((h+1)*fov_dim),height),int(w*fov_dim):min(int((w+1)*fov_dim),width)],out_range=(0,255)))
				if fov.shape[0]<fov_dim or fov.shape[1]<fov_dim:
					if black_background:
						background=np.zeros((fov_dim,fov_dim,3),dtype='uint8')
					else:
						background=np.uint8(np.ones((fov_dim,fov_dim,3),dtype='uint8')*255)
					background[:fov.shape[0],:fov.shape[1]]=fov
					fov=background
				cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'.jpg'),cv2.cvtColor(fov,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,100])

	elif os.path.splitext(os.path.basename(path_to_file))[1] in ['.tif','.TIF','.tiff','.TIFF']:

		image=imread(path_to_file)

		width=image.shape[1]
		height=image.shape[0]
		if len(list(image.shape))<3:
			c_list=None
		else:
			c_list=list(range(image.shape[2]))
		num_w=int(width/fov_dim)
		if width%fov_dim!=0:
			num_w+=1
		num_h=int(height/fov_dim)
		if height%fov_dim!=0:
			num_h+=1

		for h in range(num_h):

			for w in range(num_w):

				if c_list is None:
					fov=np.uint8(exposure.rescale_intensity(image[int(h*fov_dim):min(int((h+1)*fov_dim),height),int(w*fov_dim):min(int((w+1)*fov_dim),width)],out_range=(0,255)))
					if fov.shape[0]<fov_dim or fov.shape[1]<fov_dim:
						if black_background:
							background=np.zeros((fov_dim,fov_dim),dtype='uint8')
						else:
							background=np.uint8(np.ones((fov_dim,fov_dim),dtype='uint8')*255)
						background[:fov.shape[0],:fov.shape[1]]=fov
						fov=background
					cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'.jpg'),cv2.cvtColor(fov,cv2.COLOR_GRAY2BGR),[cv2.IMWRITE_JPEG_QUALITY,100])
				else:
					for c in c_list:
						fov=np.uint8(exposure.rescale_intensity(image[int(h*fov_dim):min(int((h+1)*fov_dim),height),int(w*fov_dim):min(int((w+1)*fov_dim),width),c],out_range=(0,255)))
						if fov.shape[0]<fov_dim or fov.shape[1]<fov_dim:
							if black_background:
								background=np.zeros((fov_dim,fov_dim,3),dtype='uint8')
							else:
								background=np.uint8(np.ones((fov_dim,fov_dim,3),dtype='uint8')*255)
							background[:fov.shape[0],:fov.shape[1]]=fov
							fov=background
						cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'_c'+str(c)+'.jpg'),cv2.cvtColor(fov,cv2.COLOR_GRAY2BGR),[cv2.IMWRITE_JPEG_QUALITY,100])

	elif os.path.splitext(os.path.basename(path_to_file))[1] in ['.qptiff','.QPTIFF']:

		image=imread(path_to_file)

		c_list=list(range(image.shape[0]))
		width=image.shape[2]
		height=image.shape[1]
		num_w=int(width/fov_dim)
		if width%fov_dim!=0:
			num_w+=1
		num_h=int(height/fov_dim)
		if height%fov_dim!=0:
			num_h+=1

		for c in c_list:

			for h in range(num_h):

				for w in range(num_w):

					fov=np.uint8(exposure.rescale_intensity(image[c,int(h*fov_dim):min(int((h+1)*fov_dim),height),int(w*fov_dim):min(int((w+1)*fov_dim),width)],out_range=(0,255)))
					if fov.shape[0]<fov_dim or fov.shape[1]<fov_dim:
						if black_background:
							background=np.zeros((fov_dim,fov_dim),dtype='uint8')
						else:
							background=np.uint8(np.ones((fov_dim,fov_dim),dtype='uint8')*255)
						background[:fov.shape[0],:fov.shape[1]]=fov
						fov=background
					cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'_c'+str(c)+'.jpg'),cv2.cvtColor(fov,cv2.COLOR_GRAY2BGR),[cv2.IMWRITE_JPEG_QUALITY,100])

	else:

		lifdata=LifFile(path_to_file)
		file=[i for i in lifdata.get_iter_image()][0]

		c_list=[i for i in file.get_iter_c(t=0,z=0)]

		image=np.array(file.get_frame(z=0,t=0,c=0))
		width=image.shape[1]
		height=image.shape[0]
		num_w=int(width/fov_dim)
		if width%fov_dim!=0:
			num_w+=1
		num_h=int(height/fov_dim)
		if height%fov_dim!=0:
			num_h+=1

		for h in range(num_h):

			for w in range(num_w):

				for c in range(len(c_list)):

					image=np.array(file.get_frame(z=0,t=0,c=c))
					fov=np.uint8(exposure.rescale_intensity(image[int(h*fov_dim):min(int((h+1)*fov_dim),height),int(w*fov_dim):min(int((w+1)*fov_dim),width)],out_range=(0,255)))
					if fov.shape[0]<fov_dim or fov.shape[1]<fov_dim:
						if black_background:
							background=np.zeros((fov_dim,fov_dim),dtype='uint8')
						else:
							background=np.uint8(np.ones((fov_dim,fov_dim),dtype='uint8')*255)
						background[:fov.shape[0],:fov.shape[1]]=fov
						fov=background
					cv2.imwrite(os.path.join(out_folder,os.path.splitext(os.path.basename(path_to_file))[0]+'_'+str(w)+str(h)+'_c'+str(c)+'.jpg'),cv2.cvtColor(fov,cv2.COLOR_GRAY2BGR),[cv2.IMWRITE_JPEG_QUALITY,100])


def preprocess_image(path_to_image,out_folder,downsize_factor,enhance_contrast=True,contrast=1.0,crop_image=True,left=0,right=0,top=0,bottom=0,gray_scale=False):

	name=os.path.basename(path_to_image).split('.')[0]
	extension=os.path.splitext(os.path.basename(path_to_image))[1]

	image=imread(path_to_image)

	if extension in ['.qptiff','.QPTIFF']:
		width=image.shape[2]
		height=image.shape[1]
	else:
		width=image.shape[1]
		height=image.shape[0]

	if gray_scale:
		if extension in ['.svs','.SVS']:
			image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
			image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
	
	if downsize_factor is not None:
		if extension in ['.qptiff','.QPTIFF']:
			image=cv2.resize(image,(int(width*downsize_factor/100),int(height*downsize_factor/100)),interpolation=cv2.INTER_AREA)
		else:
			image=cv2.resize(image,(int(width*downsize_factor/100),int(height*downsize_factor/100)),interpolation=cv2.INTER_AREA)

	if crop_image:
		image=image[top:bottom,left:right,:]

	if enhance_contrast:
		image=image*contrast
		image[image>255]=255

	image=np.uint8(image)

	if extension in ['.svs','.SVS']:
		imwrite(os.path.join(out_folder,name+'_processed.tif'),image,photometric='rgb')
	else:
		imwrite(os.path.join(out_folder,name+'_processed.tif'),image,photometric='rgb')

	print('The processed image(s) stored in: '+out_folder)


def calculate_totalintensity(path_to_file,results_path):

	images={}
	basename=os.path.splitext(os.path.basename(path_to_file))
	os.makedirs(os.path.join(results_path,basename[0]),exist_ok=True)

	if basename[1] in ['.lif','.LIF']:
		lifdata=LifFile(path_to_file)
		file=[i for i in lifdata.get_iter_image()][0]
		for c in range(len([i for i in file.get_iter_c(t=0,z=0)])):
			images[c]=np.array(file.get_frame(z=0,t=0,c=c))
	else:
		if basename[1] in ['.qptiff','.QPTIFF']:
			for c in list(range(imread(path_to_file).shape[0])):
				images[c]=imread(path_to_file)[c,:,:]
		else:
			for c in list(range(imread(path_to_file).shape[2])):
				images[c]=imread(path_to_file)[:,:,c]

	areas={}
	intensities={}

	for c in images:
		binary_mask=images[c]>0
		area=np.count_nonzero(binary_mask)
		areas[c]=area
		if area>0:
			intensities[c]=np.sum(images[c][binary_mask])/area
		else:
			intensities[c]=0

	dfs={}
	for c in images:
		dfs['area_'+str(c)]=areas[c]
		dfs['intensity_'+str(c)]=intensities[c]
	dfs=pd.DataFrame(dfs,index=['value'])
	out_sheet=os.path.join(results_path,basename[0],basename[0]+'_total_intensity.xlsx')
	dfs.to_excel(out_sheet,float_format='%.2f')

	print('Analysis completed!')



