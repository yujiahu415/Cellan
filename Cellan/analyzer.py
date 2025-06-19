from .detector import Detector
import os
import cv2
import json
import torch
import datetime
import numpy as np
import pandas as pd
from readlif.reader import LifFile
from tifffile import imread,imwrite
from skimage import exposure



class AnalyzeCells():

	def __init__(self,path_to_file,results_path,path_to_detector,cell_kinds,names_colors,detection_threshold=None,expansion=None,show_ids=False,filters={},inners=None):

		self.path_to_file=path_to_file
		self.results_path=os.path.join(results_path,os.path.splitext(os.path.basename(self.path_to_file))[0])
		if os.path.splitext(os.path.basename(self.path_to_file))[1] in ['.lif','.LIF']:
			self.lif=True
		else: 
			self.lif=False
		os.makedirs(self.results_path,exist_ok=True)
		self.detector=Detector()
		self.detector.load(path_to_detector,cell_kinds)
		self.cell_kinds=cell_kinds
		self.cell_mapping=self.detector.cell_mapping
		self.names_colors=names_colors
		self.detection_threshold=detection_threshold
		if self.detection_threshold is None:
			self.detection_threshold={}
			for cell_name in self.cell_kinds:
				self.detection_threshold[cell_name]=0
		self.expansion=expansion
		self.fov_dim=self.detector.inferencing_framesize
		self.black_background=self.detector.black_background
		self.show_ids=show_ids
		self.filters=filters
		self.inners=inners


	def analyze_multichannels(self,detection_channel=0,analysis_channels=[]):

		if self.lif:
			lifdata=LifFile(self.path_to_file)
			file=[i for i in lifdata.get_iter_image()][0]
			detect_image=np.array(file.get_frame(z=0,t=0,c=detection_channel))
			if len(analysis_channels)==0:
				c_list=[i for i in file.get_iter_c(t=0,z=0)]
				analysis_channels=list(range(len(c_list)))
		else:
			if os.path.splitext(os.path.basename(self.path_to_file))[1] in ['.qptiff','.QPTIFF']:
				detect_image=imread(self.path_to_file)[detection_channel,:,:]
				if len(analysis_channels)==0:
					analysis_channels=list(range(imread(self.path_to_file).shape[0]))
			else:
				detect_image=imread(self.path_to_file)[:,:,detection_channel]
				if len(analysis_channels)==0:
					analysis_channels=list(range(imread(self.path_to_file).shape[2]))

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

		detect_image=cv2.cvtColor(np.uint8(detect_image),cv2.COLOR_GRAY2BGR)
		width=detect_image.shape[1]
		height=detect_image.shape[0]
		num_w=int(width/self.fov_dim)
		if width%self.fov_dim!=0:
			num_w+=1
		num_h=int(height/self.fov_dim)
		if height%self.fov_dim!=0:
			num_h+=1

		thickness=max(1,round(self.fov_dim/960))

		for h in range(num_h):

			for w in range(num_w):

				detect_fov=np.uint8(exposure.rescale_intensity(detect_image[int(h*self.fov_dim):min(int((h+1)*self.fov_dim),height),int(w*self.fov_dim):min(int((w+1)*self.fov_dim),width)],out_range=(0,255)))
				if detect_fov.shape[0]<self.fov_dim or detect_fov.shape[1]<self.fov_dim:
					if self.black_background:
						background=np.zeros((self.fov_dim,self.fov_dim),dtype='uint8')
					else:
						background=np.uint8(np.ones((self.fov_dim,self.fov_dim),dtype='uint8')*255)
					background[:detect_fov.shape[0],:detect_fov.shape[1]]=detect_fov
					detect_fov=background

				analysis_fovs={}
				to_annotates={}
				for c in analysis_channels:
					if self.lif:
						analysis_fov=np.array(file.get_frame(z=0,t=0,c=c))[int(h*self.fov_dim):min(int((h+1)*self.fov_dim),height),int(w*self.fov_dim):min(int((w+1)*self.fov_dim),width)]
					else:
						if os.path.splitext(os.path.basename(self.path_to_file))[1] in ['.qptiff','.QPTIFF']:
							analysis_fov=imread(self.path_to_file)[c,int(h*self.fov_dim):min(int((h+1)*self.fov_dim),height),int(w*self.fov_dim):min(int((w+1)*self.fov_dim),width)]
						else:
							analysis_fov=imread(self.path_to_file)[int(h*self.fov_dim):min(int((h+1)*self.fov_dim),height),int(w*self.fov_dim):min(int((w+1)*self.fov_dim),width),c]
					if analysis_fov.shape[0]<self.fov_dim or analysis_fov.shape[1]<self.fov_dim:
						if self.black_background:
							background=np.zeros((self.fov_dim,self.fov_dim),dtype='uint8')
						else:
							background=np.uint8(np.ones((self.fov_dim,self.fov_dim),dtype='uint8')*255)
						background[:analysis_fov.shape[0],:analysis_fov.shape[1]]=analysis_fov
						analysis_fov=background
					analysis_fovs[c]=analysis_fov
					to_annotates[c]=cv2.cvtColor(np.uint8(exposure.rescale_intensity(analysis_fov,out_range=(0,255))),cv2.COLOR_GRAY2BGR)

				output=self.detector.inference([{'image':torch.as_tensor(detect_fov.astype('float32').transpose(2,0,1))}])
				instances=output[0]['instances'].to('cpu')
				masks=instances.pred_masks.numpy().astype(np.uint8)
				classes=instances.pred_classes.numpy()
				classes=[self.cell_mapping[str(x)] for x in classes]
				scores=instances.scores.numpy()

				if len(masks)>0:

					for cell_name in self.cell_kinds:

						hex_color=self.names_colors[cell_name].lstrip('#')
						color=tuple(int(hex_color[i:i+2],16) for i in (0,2,4))
						color=color[::-1]

						cell_masks=[masks[a] for a,name in enumerate(classes) if name==cell_name]
						cell_scores=[scores[a] for a,name in enumerate(classes) if name==cell_name]

						if len(cell_masks)>0:

							mask_area=np.sum(np.array(cell_masks),axis=(1,2))
							exclusion_mask=np.zeros(len(cell_masks),dtype=bool)
							exclusion_mask[np.where((np.sum(np.logical_and(np.array(cell_masks)[:,None],cell_masks),axis=(2,3))/mask_area[:,None]>0.8) & (mask_area[:,None]<mask_area[None,:]))[0]]=True
							cell_masks=[m for m,exclude in zip(cell_masks,exclusion_mask) if not exclude]
							cell_scores=[s for s,exclude in zip(cell_scores,exclusion_mask) if not exclude]

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
											cx=int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00'])+int(w*self.fov_dim)
											cy=int(cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00'])+int(h*self.fov_dim)
											cell_centers[cell_name].append((cx,cy))
											cell_areas[cell_name].append(np.sum(np.array(mask),axis=(0,1)))

										for c in analysis_channels:
											analysis_fov=analysis_fovs[c]
											to_annotate=to_annotates[c]
											for n,cnt in enumerate(goodcontours):
												area=cell_areas[cell_name][n]
												if area>0:
													cell_intensities[cell_name][c].append(np.sum(analysis_fov*goodmasks[n])/area)
													cv2.drawContours(to_annotate,[cnt],0,color,thickness)
													if self.show_ids:
														cx-=int(w*self.fov_dim)
														cy-=int(h*self.fov_dim)
														cv2.putText(to_annotate,str(len(cell_intensities[cell_name][c])),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,thickness,color,thickness)
													to_annotates[c]=to_annotate
												else:
													cell_intensities[cell_name][c].append(0)

					for c in analysis_channels:
						cv2.imwrite(os.path.join(self.results_path,os.path.splitext(os.path.basename(self.path_to_file))[0]+'_'+str(w)+str(h)+'_c'+str(c)+'_annotated.jpg'),to_annotates[c])

		for cell_name in self.cell_kinds:

			dfs=[]

			if len(cell_centers[cell_name])>0:
				dfs.append(pd.DataFrame([i+1 for i in range(len(cell_centers[cell_name]))],columns=['number']).reset_index(drop=True))
				dfs.append(pd.DataFrame(cell_centers[cell_name],columns=['center_x','center_y']).reset_index(drop=True))
				dfs.append(pd.DataFrame(cell_areas[cell_name],columns=['areas']).reset_index(drop=True))
				for c in analysis_channels:
					dfs.append(pd.DataFrame(cell_intensities[cell_name][c],columns=['intensity_'+str(c)]).reset_index(drop=True))
			else:
				dfs.append(pd.DataFrame(['NA'],columns=['number']).reset_index(drop=True))
				dfs.append(pd.DataFrame([('NA','NA')],columns=['center_x','center_y']).reset_index(drop=True))
				dfs.append(pd.DataFrame(['NA'],columns=['areas']).reset_index(drop=True))
				for c in analysis_channels:
					dfs.append(pd.DataFrame(['NA'],columns=['intensity_'+str(c)]).reset_index(drop=True))
			out_sheet=os.path.join(self.results_path,os.path.splitext(os.path.basename(self.path_to_file))[0]+'_'+cell_name+'_summary.xlsx')
			pd.concat(dfs,axis=1).to_excel(out_sheet,float_format='%.2f',index_label='ID/parameter')

		print('Analysis completed!')


	def analyze_singlechannel(self):

		data={}
		annotation={'segmentations':[],'class_names':[]}
		total_foreground_area=0
		parameters=['center','area','height','width','perimeter','roundness','intensity']

		for cell_name in self.cell_kinds:
			data[cell_name]={}
			data[cell_name]['total_cell_area']=0
			for parameter in parameters:
				data[cell_name][parameter]=[]

		image_name=os.path.basename(self.path_to_file)
		if os.path.splitext(image_name)[1] in ['.jpg','.JPG','.png','.PNG']:
			image=cv2.imread(self.path_to_file)
		else:
			image=imread(self.path_to_file)
			image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
		image_name=os.path.basename(self.path_to_file)
		width=image.shape[1]
		height=image.shape[0]
		num_w=int(width/self.fov_dim)
		if width%self.fov_dim!=0:
			num_w+=1
		num_h=int(height/self.fov_dim)
		if height%self.fov_dim!=0:
			num_h+=1

		to_annotate=np.uint8(exposure.rescale_intensity(image,out_range=(0,255)))
		thickness=max(1,round(self.fov_dim/960))

		for h in range(num_h):

			for w in range(num_w):

				offset=np.array([[[int(w*self.fov_dim),int(h*self.fov_dim)]]])

				analysis_fov=image[int(h*self.fov_dim):min(int((h+1)*self.fov_dim),height),int(w*self.fov_dim):min(int((w+1)*self.fov_dim),width)]
				detect_fov=np.uint8(exposure.rescale_intensity(analysis_fov,out_range=(0,255)))
				if detect_fov.shape[0]<self.fov_dim or detect_fov.shape[1]<self.fov_dim:
					if self.black_background:
						background_analysis=np.zeros((self.fov_dim,self.fov_dim,3),dtype='uint8')
						background_detect=np.zeros((self.fov_dim,self.fov_dim,3),dtype='uint8')
					else:
						background_analysis=np.uint8(np.ones((self.fov_dim,self.fov_dim,3),dtype='uint8')*255)
						background_detect=np.uint8(np.ones((self.fov_dim,self.fov_dim,3),dtype='uint8')*255)
					background_analysis[:detect_fov.shape[0],:detect_fov.shape[1]]=analysis_fov
					background_detect[:detect_fov.shape[0],:detect_fov.shape[1]]=detect_fov
					analysis_fov=background_analysis
					detect_fov=background_detect
				if self.black_background:
					area_noholes=np.count_nonzero(cv2.threshold(cv2.cvtColor(detect_fov,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
				else:
					area_noholes=detect_fov.shape[0]*detect_fov.shape[1]-np.count_nonzero(cv2.threshold(cv2.cvtColor(detect_fov,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])

				total_foreground_area+=area_noholes

				output=self.detector.inference([{'image':torch.as_tensor(detect_fov.astype('float32').transpose(2,0,1))}])
				instances=output[0]['instances'].to('cpu')
				masks=instances.pred_masks.numpy().astype(np.uint8)
				classes=instances.pred_classes.numpy()
				classes=[self.cell_mapping[str(x)] for x in classes]
				scores=instances.scores.numpy()

				if len(masks)>0:

					for cell_name in self.cell_kinds:

						hex_color=self.names_colors[cell_name].lstrip('#')
						color=tuple(int(hex_color[i:i+2],16) for i in (0,2,4))
						color=color[::-1]

						cell_masks=[masks[a] for a,name in enumerate(classes) if name==cell_name]
						cell_scores=[scores[a] for a,name in enumerate(classes) if name==cell_name]

						if len(cell_masks)>0:

							mask_area=np.sum(np.array(cell_masks),axis=(1,2))
							exclusion_mask=np.zeros(len(cell_masks),dtype=bool)
							exclusion_mask[np.where((np.sum(np.logical_and(np.array(cell_masks)[:,None],cell_masks),axis=(2,3))/mask_area[:,None]>0.8) & (mask_area[:,None]<mask_area[None,:]))[0]]=True
							cell_masks=[m for m,exclude in zip(cell_masks,exclusion_mask) if not exclude]
							cell_scores=[s for s,exclude in zip(cell_scores,exclusion_mask) if not exclude]

							if len(cell_masks)>0:

									goodmasks=[cell_masks[x] for x,score in enumerate(cell_scores) if score>=self.detection_threshold[cell_name]]

									if len(goodmasks)>0:

										for mask in goodmasks:
											mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
											if self.expansion is not None:
												mask=cv2.dilate(mask,np.ones((5,5),np.uint8),iterations=self.expansion)
											cnts,_=cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
											if len(cnts)>0:
												cnt=sorted(cnts,key=cv2.contourArea,reverse=True)[0]
												area=np.sum(np.array(mask),axis=(0,1))
												perimeter=cv2.arcLength(cnt,closed=True)
												roundness=(4*np.pi*area)/(perimeter*perimeter)
												(_,_),(wd,ht),_=cv2.minAreaRect(cnt)
												intensity=np.sum(analysis_fov*cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR))/max(area,1)
												cnt=cnt+offset
												segmentation=cnt.flatten().tolist()
												if 'area' in self.filters:
													if area<self.filters['area'][0] or area>self.filters['area'][1]:
														continue
												if 'perimeter' in self.filters:
													if perimeter<self.filters['perimeter'][0] or perimeter>self.filters['perimeter'][1]:
														continue
												if 'roundness' in self.filters:
													if roundness<self.filters['roundness'][0] or roundness>self.filters['roundness'][1]:
														continue
												if 'height' in self.filters:
													if ht<self.filters['height'][0] or ht>self.filters['height'][1]:
														continue
												if 'width' in self.filters:
													if wd<self.filters['width'][0] or wd>self.filters['width'][1]:
														continue
												if area>0:
													cx=int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00'])+int(w*self.fov_dim)
													cy=int(cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00'])+int(h*self.fov_dim)
													data[cell_name]['center'].append((cx,cy))
													data[cell_name]['area'].append(area)
													data[cell_name]['height'].append(ht)
													data[cell_name]['width'].append(wd)
													data[cell_name]['perimeter'].append(perimeter)
													data[cell_name]['roundness'].append(roundness)
													data[cell_name]['intensity'].append(intensity)
													annotation['segmentations'].append(segmentation)
													annotation['class_names'].append(cell_name)
													cv2.drawContours(to_annotate,[cnt],0,color,thickness)
													if self.show_ids:
														cv2.putText(to_annotate,str(len(data[cell_name]['center'])),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,thickness,color,thickness)
													data[cell_name]['total_cell_area']+=area

		cv2.imwrite(os.path.join(self.results_path,os.path.splitext(image_name)[0]+'_annotated.'+image_name.split('.')[-1]),to_annotate)

		with pd.ExcelWriter(os.path.join(self.results_path,os.path.splitext(image_name)[0]+'_summary.xlsx'),engine='openpyxl') as writer:

			for cell_name in self.cell_kinds:

				rows=[]
				columns=['filename','ID']+parameters

				if cell_name in data:
					values=zip(*[data[cell_name][parameter] for parameter in parameters])
					for idx,value in enumerate(values):
						rows.append([os.path.splitext(image_name)[0],idx+1]+list(value))

				df=pd.DataFrame(rows,columns=columns)
				df.to_excel(writer,sheet_name=cell_name,float_format='%.2f',index=False)

		with pd.ExcelWriter(os.path.join(self.results_path,os.path.splitext(image_name)[0]+'_arearatio.xlsx'),engine='openpyxl') as writer:

			for cell_name in self.cell_kinds:

				dfs={}
				dfs['total_area']=total_foreground_area
				dfs[cell_name+'_area']=data[cell_name]['total_cell_area']
				dfs['area_ratio']=data[cell_name]['total_cell_area']/total_foreground_area
				dfs=pd.DataFrame(dfs,index=['value'])
				dfs.to_excel(writer,sheet_name=cell_name,float_format='%.6f')

		coco_format={'categories':[],'images':[],'annotations':[]}

		for i,cell_name in enumerate(self.cell_kinds):
			coco_format['categories'].append({
				'id':i+1,
				'name':cell_name,
				'supercategory':'none'})

		annotation_id=0
		imwrite(os.path.join(self.results_path,image_name),cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

		coco_format['images'].append({
			'id':0,
			'width':image.shape[1],
			'height':image.shape[0],
			'file_name':image_name})

		for i,seg in enumerate(annotation['segmentations']):

			category_id=self.cell_kinds.index(annotation['class_names'][i])+1
			polygon=[(seg[x],seg[x+1]) for x in range(0,len(seg)-1,2)]

			n=len(polygon)
			area=0
			for i in range(n):
				x1,y1=polygon[i]
				x2,y2=polygon[(i+1)%n]
				area+=x1*y2-x2*y1
			area=abs(area)/2

			x_coords,y_coords=zip(*polygon)
			x_min=int(min(x_coords))
			y_min=int(min(y_coords))
			x_max=int(max(x_coords))
			y_max=int(max(y_coords))
			bbox=[x_min,y_min,x_max-x_min,y_max-y_min]

			coco_format['annotations'].append({
				'id':annotation_id,
				'image_id':0,
				'category_id':category_id,
				'segmentation':[seg],
				'area':area,
				'bbox':bbox,
				'iscrowd':0
				})

			annotation_id+=1

		with open(os.path.join(self.results_path,'annotations.json'),'w') as json_file:
			json.dump(coco_format,json_file)

		print('Analysis completed!')


