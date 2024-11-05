from .detector import Detector
import os
import cv2
import torch
import datetime
import numpy as np
import pandas as pd
from readlif.reader import LifFile
from tifffile import imread
from skimage import exposure



class AnalyzeCells():

	def __init__(self,path_to_file,results_path,path_to_detector,cell_kinds,detection_threshold=None,expansion=None,fov_div=1):

		self.detector=Detector()
		self.detector.load(path_to_detector,cell_kinds)
		self.cell_kinds=cell_kinds
		self.cell_mapping=self.detector.cell_mapping
		self.detection_threshold=detection_threshold
		self.path_to_file=path_to_file
		self.results_path=os.path.join(results_path,os.path.splitext(os.path.basename(self.path_to_file))[0])
		if os.path.splitext(os.path.basename(self.path_to_file))[1] in ['.tif','.TIF','.tiff','.TIFF']:
			self.tif=True
		else: 
			self.tif=False
		os.makedirs(self.results_path,exist_ok=True)
		self.expansion=expansion
		self.fov_div=fov_div


	def channels_intensity(self,names_colors,detection_channel=0,analysis_channels=[]):

		if self.tif:
			tifdata=imread(self.path_to_file)
			file=[i for i in tifdata]
		else:
			lifdata=LifFile(self.path_to_file)
			file=[i for i in lifdata.get_iter_image()][0]

		if len(analysis_channels)==0:
			c_list=[i for i in file.get_iter_c(t=0,z=0)]
			analysis_channels=[c for c in range(c_list)]

		if self.detection_threshold is None:
			self.detection_threshold={}
			for cell_name in self.cell_kinds:
				self.detection_threshold[cell_name]=0

		cell_numbers={}
		cell_centers={}
		cell_areas={}
		cell_intensities={}

		for cell_name in self.cell_kinds:
			cell_numbers[cell_name]=0
			cell_centers[cell_name]=[]
			cell_areas[cell_name]=[]
			cell_intensities[cell_name]={}
			for c in analysis_channels:
				cell_intensities[cell_name][c]=[]

		detect_image=np.array(file.get_frame(z=0,t=0,c=detection_channel))
		detect_image=cv2.cvtColor(detect_image,cv2.COLOR_GRAY2BGR)
		width=detect_image.shape[1]
		height=detect_image.shape[0]
		fov_width=int(width/self.fov_div)
		fov_height=int(height/self.fov_div)

		for w in range(self.fov_div):

			for h in range(self.fov_div):

				detect_fov=np.uint8(exposure.rescale_intensity(detect_image[h*fov_height:(h+1)*fov_height,w*fov_width:(w+1)*fov_width],out_range=(0,255)))
				analysis_fovs={}
				for c in analysis_channels:
					analysis_fovs[c]=np.array(file.get_frame(z=0,t=0,c=c))[h*fov_height:(h+1)*fov_height,w*fov_width:(w+1)*fov_width]
				output=self.detector.inference([{'image':torch.as_tensor(detect_fov.astype('float32').transpose(2,0,1))}])
				instances=output[0]['instances'].to('cpu')
				masks=instances.pred_masks.numpy().astype(np.uint8)
				classes=instances.pred_classes.numpy()
				scores=instances.scores.numpy()

				if len(masks)>0:

					mask_area=np.sum(np.array(masks),axis=(1,2))
					exclusion_mask=np.zeros(len(masks),dtype=bool)
					exclusion_mask[np.where((np.sum(np.logical_and(masks[:,None],masks),axis=(2,3))/mask_area[:,None]>0.8) & (mask_area[:,None]<mask_area[None,:]))[0]]=True
					masks=[m for m,exclude in zip(masks,exclusion_mask) if not exclude]
					classes=[c for c,exclude in zip(classes,exclusion_mask) if not exclude]
					classes=[self.cell_mapping[str(x)] for x in classes]
					scores=[s for s,exclude in zip(scores,exclusion_mask) if not exclude]

					for cell_name in self.cell_kinds:

						cell_masks=[masks[a] for a,name in enumerate(classes) if name==cell_name]
						cell_scores=[scores[a] for a,name in enumerate(classes) if name==cell_name]

						if len(cell_masks)>0:

								goodmasks=[cell_masks[x] for x,score in enumerate(cell_scores) if score>=self.detection_threshold[cell_name]]
								goodcontours=[]

								if len(goodmasks)>0:

									cell_numbers[cell_name]+=len(goodmasks)

									for mask in goodmasks:
										mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
										if self.expansion is not None:
											mask=cv2.dilate(mask,np.ones((5,5),np.uint8),iterations=self.expansion)
										cnts,_=cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
										cnt=sorted(cnts,key=cv2.contourArea,reverse=True)[0]
										goodcontours.append(cnt)
										cell_centers[cell_name].append((int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00'])+int(w*fov_width),int(cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00'])+int(h*fov_height)))
										cell_areas[cell_name].append(np.sum(np.array(mask),axis=(0,1)))

									for c in analysis_channels:
										analysis_fov=analysis_fovs[c]
										to_annotate=np.uint8(exposure.rescale_intensity(analysis_fov,out_range=(0,255)))
										for n,cnt in enumerate(goodcontours):
											area=cell_areas[cell_name][n]
											if area>0:
												cell_intensities[cell_name][c].append(np.sum(analysis_fov*goodmasks[n])/area)
												cv2.drawContours(to_annotate,[cnt],0,names_colors[cell_name],1)
											else:
												cell_intensities[cell_name][c].append(0)
										cv2.imwrite(os.path.join(self.results_path,os.path.splitext(os.path.basename(self.path_to_file))[0]+'_'+str(w)+str(h)+'_c'+str(c)+'_annotated.jpg'),to_annotate)

		for cell_name in self.cell_kinds:

			dfs=[]

			dfs.append(pd.DataFrame([i+1 for i in range(len(cell_centers[cell_name]))],columns=['number']).reset_index(drop=True))
			dfs.append(pd.DataFrame(cell_centers[cell_name],columns=['center_x','center_y']).reset_index(drop=True))
			dfs.append(pd.DataFrame(cell_areas[cell_name],columns=['areas']).reset_index(drop=True))
			for c in analysis_channels:
				dfs.append(pd.DataFrame(cell_intensities[cell_name][c],columns=['intensity_'+str(c)]).reset_index(drop=True))

			out_sheet=os.path.join(self.results_path,cell_name+'_summary.xlsx')
			pd.concat(dfs,axis=1).to_excel(out_sheet,float_format='%.2f',index_label='ID/parameter')


		print('Analysis completed!')




