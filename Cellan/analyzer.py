from .detector import Detector
import os
import cv2
import json
import torch
import datetime
import numpy as np
import pandas as pd
from scipy.spatial import distance
from readlif.reader import LifFile
from tifffile import imread,imwrite
from skimage import exposure



class AnalyzeCells():

	def __init__(self,path_to_file,results_path,path_to_detector,cell_kinds,names_colors,detection_threshold=None,expansion=None,show_ids=False,filters={}):

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

		coco_format={'info':{'year':'','version':'1','description':'Cellan annotations','contributor':'','url':'https://github.com/yujiahu415/Cellan','date_created':''},'licenses':[],'categories':[],'images':[],'annotations':[]}

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



class AnalyzeCalciumSignal():

	def __init__(self,path_to_lif,results_path,stim_t,duration):

		self.detector=None
		self.neuro_mapping=None
		self.path_to_lif=path_to_lif
		self.results_path=os.path.join(results_path,os.path.splitext(os.path.basename(self.path_to_lif))[0])
		if os.path.splitext(os.path.basename(self.path_to_lif))[1] in ['.tif','.TIF','.tiff','.TIFF']:
			self.tif=True
		else:
			self.tif=False
		os.makedirs(self.results_path,exist_ok=True)
		self.neuro_number=None
		self.neuro_kinds=None  # the catgories of neural structures to be analyzed
		self.stim_t=stim_t  # the frame number when stimulation is on
		if self.tif:
			tifdata=imread(self.path_to_lif)
			self.full_duration=tifdata.shape[0]
		else:
			lifdata=LifFile(self.path_to_lif)
			file=[i for i in lifdata.get_iter_image()][0]
			self.full_duration=len([i for i in file.get_iter_t(c=0,z=0)])-1
		self.duration=duration # the duration (in frames) for example generation / analysis, 0: use entire duration
		if self.duration<=0:
			self.duration=self.full_duration
		self.main_channel=0 # main_channel: the channel for frames to analyze
		self.to_deregister={}
		self.register_counts={}
		self.neuro_contours={}
		self.neuro_centers={}
		self.neuro_existingcenters={}
		self.neuro_masks={}
		self.neuro_areas={}
		self.neuro_Fmeans={}
		self.neuro_correctFmeans={}
		self.all_parameters={}


	def prepare_analysis(self,path_to_detector,neuro_number,neuro_kinds):
		
		print('Preparation started...')
		print(datetime.datetime.now())

		self.detector=Detector()
		self.detector.load(path_to_detector,neuro_kinds)
		self.neuro_mapping=self.detector.neuro_mapping
		self.neuro_number=neuro_number
		self.neuro_kinds=neuro_kinds

		total_number=0

		for neuro_name in self.neuro_kinds:

			total_number+=self.neuro_number[neuro_name]
			self.all_parameters[neuro_name]={}
			for parameter_name in ['F0','dF/F0','Fmax','Stim_t']:
				self.all_parameters[neuro_name][parameter_name]={}
			self.to_deregister[neuro_name]={}
			self.register_counts[neuro_name]={}
			self.neuro_contours[neuro_name]={}
			self.neuro_centers[neuro_name]={}
			self.neuro_existingcenters[neuro_name]={}
			self.neuro_masks[neuro_name]={}
			self.neuro_areas[neuro_name]={}
			self.neuro_Fmeans[neuro_name]={}
			self.neuro_correctFmeans[neuro_name]={}

			for i in range(self.neuro_number[neuro_name]):
				self.to_deregister[neuro_name][i]=0
				self.register_counts[neuro_name][i]=None
				self.neuro_contours[neuro_name][i]=[None]*self.duration
				self.neuro_centers[neuro_name][i]=[None]*self.duration
				self.neuro_existingcenters[neuro_name][i]=(-10000,-10000)
				self.neuro_masks[neuro_name][i]=[None]*self.duration
				self.neuro_areas[neuro_name][i]=[None]*self.duration
				self.neuro_Fmeans[neuro_name][i]=[0.0]*self.duration
				self.neuro_correctFmeans[neuro_name][i]=[0.0]*self.duration

		print('Preparation completed!')


	def track_neuro(self,frame_count,neuro_name,contours,centers,masks,areas,Fmeans):

		unused_existing_indices=list(self.neuro_existingcenters[neuro_name])
		existing_centers=list(self.neuro_existingcenters[neuro_name].values())
		unused_new_indices=list(range(len(centers)))
		dt_flattened=distance.cdist(existing_centers,centers).flatten()
		dt_sort_index=dt_flattened.argsort()
		length=len(centers)

		for idx in dt_sort_index:
			index_in_existing=int(idx/length)
			index_in_new=int(idx%length)
			if self.neuro_existingcenters[neuro_name][index_in_existing][0]==-10000:
				dt=np.inf
			else:
				dt=50
			if dt_flattened[idx]<dt:
				if index_in_existing in unused_existing_indices:
					if index_in_new in unused_new_indices:
						unused_existing_indices.remove(index_in_existing)
						unused_new_indices.remove(index_in_new)
						if self.register_counts[neuro_name][index_in_existing] is None:
							self.register_counts[neuro_name][index_in_existing]=frame_count
						self.to_deregister[neuro_name][index_in_existing]=0
						self.neuro_contours[neuro_name][index_in_existing][frame_count]=contours[index_in_new]
						center=centers[index_in_new]
						self.neuro_centers[neuro_name][index_in_existing][frame_count]=center
						self.neuro_existingcenters[neuro_name][index_in_existing]=center
						self.neuro_masks[neuro_name][index_in_existing][frame_count]=masks[index_in_new]
						self.neuro_areas[neuro_name][index_in_existing][frame_count]=areas[index_in_new]
						self.neuro_Fmeans[neuro_name][index_in_existing][frame_count]=Fmeans[index_in_new]

		'''
		if len(unused_existing_indices)>0:
			for i in unused_existing_indices:
				if self.to_deregister[neuro_name][i]<5:
					self.to_deregister[neuro_name][i]+=1
				else:
					self.neuro_existingcenters[neuro_name][i]=(-10000,-10000)
		'''


	def detect_neuro(self,frames,images,batch_size,frame_count):

		# frames: frames averageprojected along z, with pixel values in float
		# images: unit8 format of frames averageprojected along z, for Detectors to detect neural structures

		tensor_images=[torch.as_tensor(image.astype("float32").transpose(2,0,1)) for image in images]
		inputs=[{"image":tensor_image} for tensor_image in tensor_images]

		outputs=self.detector.inference(inputs)

		for batch_count,output in enumerate(outputs):

			image=images[batch_count]
			frame=frames[batch_count]
			instances=outputs[batch_count]['instances'].to('cpu')
			masks=instances.pred_masks.numpy().astype(np.uint8)
			classes=instances.pred_classes.numpy()
			scores=instances.scores.numpy()

			if len(masks)>0:

				mask_area=np.sum(np.array(masks),axis=(1,2))
				exclusion_mask=np.zeros(len(masks),dtype=bool)
				exclusion_mask[np.where((np.sum(np.logical_and(masks[:,None],masks),axis=(2,3))/mask_area[:,None]>0.8) & (mask_area[:,None]<mask_area[None,:]))[0]]=True
				masks=[m for m,exclude in zip(masks,exclusion_mask) if not exclude]
				classes=[c for c,exclude in zip(classes,exclusion_mask) if not exclude]
				classes=[self.neuro_mapping[str(x)] for x in classes]
				scores=[s for s,exclude in zip(scores,exclusion_mask) if not exclude]

				for neuro_name in self.neuro_kinds:

					contours=[]
					centers=[]
					goodcontours=[]
					goodmasks=[]
					final_masks=[]
					final_areas=[]
					Fmeans=[]

					neuro_number=int(self.neuro_number[neuro_name])
					neuro_masks=[masks[a] for a,name in enumerate(classes) if name==neuro_name]
					neuro_scores=[scores[a] for a,name in enumerate(classes) if name==neuro_name]

					if len(neuro_masks)>0:

						if len(neuro_scores)>neuro_number*2:
							sorted_scores_indices=np.argsort(neuro_scores)[-int(neuro_number*2):]
							neuro_masks=[neuro_masks[x] for x in sorted_scores_indices]

						for mask in neuro_masks:
							mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
							goodmasks.append(mask)
							cnts,_=cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
							goodcontours.append(sorted(cnts,key=cv2.contourArea,reverse=True)[0])

						areas=[np.sum(np.array(m),axis=(0,1)) for m in goodmasks]
						sorted_area_indices=np.argsort(np.array(areas))[-neuro_number:]
						areas_sorted=sorted(areas)

						for x in sorted_area_indices:
							mask=goodmasks[x]
							area=areas_sorted[x]
							cnt=goodcontours[x]
							contours.append(cnt)
							centers.append((int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00']),int(cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00'])))
							final_masks.append(mask)
							final_areas.append(area)

							if area>0:
								Fmeans.append(np.sum(frame*mask)/area)
							else:
								Fmeans.append(0)

						self.track_neuro(frame_count+1-batch_size+batch_count,neuro_name,contours,centers,final_masks,final_areas,Fmeans)


	def acquire_information(self,batch_size=1,autofind_t=False,stimulation_channel=0,main_channel=0):

		# autofind_t: automatically find the frame number of stimulation, False: use stim_t
		# stimulation_channel: the channel for record stimulation onset

		print('Acquiring information in each frame...')
		print(datetime.datetime.now())

		self.main_channel=main_channel

		initial_frame=None
		stimulation_checked=False
		main_frames=[]
		images=[]
		batch_count=frame_count=0

		if self.tif:
			tifdata=imread(self.path_to_lif)
			file=[i for i in tifdata]
		else:
			lifdata=LifFile(self.path_to_lif)
			file=[i for i in lifdata.get_iter_image()][0]

		while True:

			if frame_count<self.duration:

				if self.tif:

					frame_project=np.array(file[frame_count])

				else:

					frame_project=[np.array(i) for i in file.get_iter_z(t=frame_count,c=self.main_channel)]
					#frame_project=np.array(frame_project).sum(0)/len(frame_project)
					frame_project=np.array(frame_project).max(0)

					if autofind_t is True:

						frame_project_stim=[np.array(i) for i in file.get_iter_z(t=frame_count,c=stimulation_channel)]
						frame_project_stim=np.array(frame_project_stim).sum(0)/len(frame_project_stim)

						if initial_frame is None:
							initial_frame=frame_project_stim

						if stimulation_checked is False:
							if np.mean(frame_project_stim)>1.2*np.mean(initial_frame):
								self.stim_t=frame_count
								stimulation_checked=True
								print('Stimulation onset: at frame '+str(self.stim_t)+'.')

				main_frames.append(frame_project)
				frame_project[frame_project>255]=255
				frame_project=cv2.cvtColor(np.uint8(frame_project),cv2.COLOR_GRAY2BGR)
				images.append(frame_project)

				batch_count+=1

				if batch_count==batch_size:
					batch_count=0
					self.detect_neuro(main_frames,images,batch_size,frame_count)
					main_frames=[]
					images=[]

				if (frame_count+1)%10==0:
					print(str(frame_count+1)+' frames processed...')
					print(datetime.datetime.now())

			if frame_count>=self.full_duration:
				if len(main_frames)>0:
					self.detect_neuro(main_frames,images,batch_size,frame_count)
				break

			frame_count+=1

		print('Information acquisition completed!')


	def craft_data(self):

		print('Crafting data...')
		print(datetime.datetime.now())

		for neuro_name in self.neuro_kinds:

			to_delete=[]
			IDs=list(self.neuro_centers[neuro_name].keys())

			for i in IDs:
				if self.register_counts[neuro_name][i] is None:
					to_delete.append(i)

			if len(IDs)==len(to_delete):
				print('No neural structure detected!')
			
			for i in IDs:
				if i in to_delete:
					del self.to_deregister[neuro_name][i]
					del self.register_counts[neuro_name][i]
					del self.neuro_centers[neuro_name][i]
					del self.neuro_existingcenters[neuro_name][i]
					del self.neuro_contours[neuro_name][i]
					del self.neuro_masks[neuro_name][i]
					del self.neuro_areas[neuro_name][i]
					del self.neuro_Fmeans[neuro_name][i]
					del self.neuro_correctFmeans[neuro_name][i]

			centers=[]
			contours=[]
			masks=[]
			areas=[]
			Fmeans=[]
			correctFmeans=[]

			for i in self.neuro_centers[neuro_name]:

				temp_centers=[c for c in self.neuro_centers[neuro_name][i] if c is not None]
				temp_contours=[c for c in self.neuro_contours[neuro_name][i] if c is not None]
				temp_masks=[m for m in self.neuro_masks[neuro_name][i] if m is not None]
				temp_areas=[a for a in self.neuro_areas[neuro_name][i] if a is not None]

				idx=np.argsort(temp_areas)[-1]

				centers.append(temp_centers[idx])
				contours.append(temp_contours[idx])
				masks.append(temp_masks[idx])
				areas.append(temp_areas[idx])
				Fmeans.append(self.neuro_Fmeans[neuro_name][i])
				correctFmeans.append(self.neuro_correctFmeans[neuro_name][i])

			self.neuro_centers[neuro_name]={}
			self.neuro_contours[neuro_name]={}
			self.neuro_masks[neuro_name]={}
			self.neuro_areas[neuro_name]={}
			self.neuro_Fmeans[neuro_name]={}
			self.neuro_correctFmeans[neuro_name]={}

			sorted_indices=sorted(range(len(centers)),key=lambda i:centers[i][1])
			centers=[centers[i] for i in sorted_indices]
			contours=[contours[i] for i in sorted_indices]
			masks=[masks[i] for i in sorted_indices]
			areas=[areas[i] for i in sorted_indices]
			Fmeans=[Fmeans[i] for i in sorted_indices]
			correctFmeans=[correctFmeans[i] for i in sorted_indices]

			sorted_indices=sorted(range(len(centers)),key=lambda i:centers[i][0])
			centers=[centers[i] for i in sorted_indices]
			contours=[contours[i] for i in sorted_indices]
			masks=[masks[i] for i in sorted_indices]
			areas=[areas[i] for i in sorted_indices]
			Fmeans=[Fmeans[i] for i in sorted_indices]
			correctFmeans=[correctFmeans[i] for i in sorted_indices]

			for i in range(len(sorted_indices)):
				self.neuro_centers[neuro_name][i]=centers[i]
				self.neuro_contours[neuro_name][i]=contours[i]
				self.neuro_masks[neuro_name][i]=masks[i]
				self.neuro_areas[neuro_name][i]=areas[i]
				self.neuro_Fmeans[neuro_name][i]=Fmeans[i]
				self.neuro_correctFmeans[neuro_name][i]=correctFmeans[i]

		print('Data crafting completed!')


	def annotate_video(self):

		print('Annotating video...')
		print(datetime.datetime.now())

		if self.tif:
			tifdata=imread(self.path_to_lif)
			file=[i for i in tifdata]
		else:
			lifdata=LifFile(self.path_to_lif)
			file=[i for i in lifdata.get_iter_image()][0]

		frame_count=0
		writer=None

		while True:

			if frame_count<self.duration:

				if self.tif:
					frame_project=np.array(file[frame_count])
				else:
					frame_project=[np.array(i) for i in file.get_iter_z(t=frame_count,c=self.main_channel)]
					frame_project=np.array(frame_project).sum(0)/len(frame_project)
					#frame_project=np.array(frame_project).max(0)

				frame_project[frame_project>255]=255
				frame_project=cv2.cvtColor(np.uint8(frame_project),cv2.COLOR_GRAY2BGR)

				if writer is None:
					(h,w)=frame_project.shape[:2]
					out=os.path.join(self.results_path,'Annotated video.avi')
					writer=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'MJPG'),1,(w,h),True)

				for neuro_name in self.neuro_kinds:
					for i in self.neuro_centers[neuro_name]:
						cx=self.neuro_centers[neuro_name][i][0]
						cy=self.neuro_centers[neuro_name][i][1]
						cv2.putText(frame_project,neuro_name+str(i),(cx-1,cy+1),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
						ct=self.neuro_contours[neuro_name][i]
						cv2.drawContours(frame_project,[ct],0,(0,255,0),1)

				writer.write(frame_project)

			if frame_count>=self.full_duration:
				break

			frame_count+=1

		print('Video annotation completed!')


	def quantify_parameters(self,F0_period=10,F_period=30):

		# F0_period: the duration (in frames) for calculating F0
		# F_period: the duration (in frames) for calculating dF/F0

		print('Quantifying neural activities...')
		print(datetime.datetime.now())

		frame_count=0

		if self.tif:
			tifdata=imread(self.path_to_lif)
			file=[i for i in tifdata]
		else:
			lifdata=LifFile(self.path_to_lif)
			file=[i for i in lifdata.get_iter_image()][0]

		while True:

			if frame_count<self.duration:

				if self.tif:
					frame_project=np.array(file[frame_count])
				else:
					frame_project=[np.array(i) for i in file.get_iter_z(t=frame_count,c=self.main_channel)]
					#frame_project=np.array(frame_project).sum(0)/len(frame_project)
					frame_project=np.array(frame_project).max(0)

				for neuro_name in self.neuro_kinds:

					for i in self.neuro_centers[neuro_name]:

						mask=self.neuro_masks[neuro_name][i]
						area=self.neuro_areas[neuro_name][i]

						self.neuro_correctFmeans[neuro_name][i][frame_count]=np.sum(frame_project*mask)/area

				if (frame_count+1)%10==0:
					print(str(frame_count+1)+' frames quantified...')
					print(datetime.datetime.now())

			if frame_count>=self.full_duration:
				break

			frame_count+=1

		for neuro_name in self.neuro_kinds:

			for i in self.neuro_correctFmeans[neuro_name]:

				df=pd.DataFrame(self.neuro_correctFmeans[neuro_name],index=[i for i in range(self.duration)])
				df.to_excel(os.path.join(self.results_path,neuro_name+'_F.xlsx'),float_format='%.2f',index_label='frame/ID')

				if self.stim_t<=F0_period:
					F_array=self.neuro_correctFmeans[neuro_name][i][:(self.stim_t-1)]
				else:
					F_array=self.neuro_correctFmeans[neuro_name][i][(self.stim_t-F0_period-1):(self.stim_t-1)]

				F0=np.array(F_array).mean()
				self.all_parameters[neuro_name]['F0'][i]=F0

				if self.stim_t+F_period>=self.duration:
					F_array=self.neuro_correctFmeans[neuro_name][i][self.stim_t:]
				else:
					F_array=self.neuro_correctFmeans[neuro_name][i][self.stim_t:(self.stim_t+F_period)]

				Fmax=np.array(F_array).max()
				self.all_parameters[neuro_name]['Fmax'][i]=Fmax

				if F0==0.0:
					print('The F0 of '+neuro_name+' '+str(i)+' is 0.')
					self.all_parameters[neuro_name]['dF/F0'][i]=np.nan
				else:
					self.all_parameters[neuro_name]['dF/F0'][i]=(Fmax-F0)/F0

				self.all_parameters[neuro_name]['Stim_t'][i]=self.stim_t

		parameters=[]

		for parameter_name in ['F0','Fmax','dF/F0','Stim_t']:
			df=self.all_parameters[neuro_name][parameter_name]
			parameters.append(pd.DataFrame.from_dict(df,orient='index',columns=[parameter_name]).reset_index(drop=True))

		out_sheet=os.path.join(self.results_path,neuro_name+'_summary.xlsx')
		pd.concat(parameters,axis=1).to_excel(out_sheet,float_format='%.2f',index_label='ID/parameter')

		print('All results exported in: '+str(self.results_path))


	def extract_frames(self,skip_redundant=10):

		print('Generating behavior examples...')
		print(datetime.datetime.now())


		if self.tif:
			tifdata=imread(self.path_to_lif)
			file=[i for i in tifdata]
			channels=[0]
		else:
			lifdata=LifFile(self.path_to_lif)
			file=[i for i in lifdata.get_iter_image()][0]
			channels=[i for i in file.get_iter_c(t=0,z=0)]

		end_t=self.stim_t+self.duration

		for channel in range(len(channels)):

			for frame_count in range(self.full_duration):

				if self.stim_t<=frame_count<end_t and frame_count%skip_redundant==0:

					if self.tif is True:
						frame_project=np.array(file[frame_count])
					else:
						frame_project=[np.array(i) for i in file.get_iter_z(t=frame_count,c=channel)]
						#frame_project=np.array(frame_project).sum(0)/len(frame_project)
						frame_project=np.array(frame_project).max(0)

					frame_project[frame_project>255]=255
					out_image=os.path.join(self.results_path,str(channel)+'_'+str(frame_count)+'.jpg')
					cv2.imwrite(out_image,np.uint8(np.array(frame_project)))

		print('The images stored in: '+self.results_path)

