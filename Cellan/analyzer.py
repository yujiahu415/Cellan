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
		total_foreground_area=0
		parameters=['center','area','height','width','perimeter','roundness','intensity','segmentation']

		for cell_name in self.cell_kinds:
			data[cell_name]={}
			data[cell_name][total_cell_area]=0
			for parameter in parameters:
				data[cell_name][parameter]=[]

		image=imread(self.path_to_file)
		image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
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

					to_annotate=np.uint8(exposure.rescale_intensity(analysis_fov,out_range=(0,255)))

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
													cell_numbers[cell_name]+=1
													cx=int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00'])+int(w*self.fov_dim)
													cy=int(cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00'])+int(h*self.fov_dim)
													cell_centers[cell_name].append((cx,cy))
													cell_areas[cell_name].append(area)
													cell_heights[cell_name].append(ht)
													cell_widths[cell_name].append(wd)
													cell_perimeter[cell_name].append(perimeter)
													cell_roundness[cell_name].append(roundness)
													cell_intensities[cell_name].append(intensity)
													cv2.drawContours(to_annotate,[cnt],0,color,thickness)
													if self.show_ids:
														cx-=int(w*self.fov_dim)
														cy-=int(h*self.fov_dim)
														cv2.putText(to_annotate,str(len(cell_centers[cell_name])),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,thickness,color,thickness)
													total_cell_area[cell_name]+=area
													if self.inners is not None:
														if self.inners=='white':
															thred=cv2.threshold(cv2.cvtColor(detect_fov,cv2.COLOR_BGR2GRAY)*mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
														else:
															thred=cv2.threshold(cv2.cvtColor(detect_fov,cv2.COLOR_BGR2GRAY)*mask,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
														cnts,_=cv2.findContours(thred,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
														if len(cnts)>1:
															cnt=sorted(cnts,key=cv2.contourArea,reverse=True)[1]
															cv2.drawContours(to_annotate,[cnt],0,color,thickness)
															area=np.count_nonzero(thred)
															(_,_),(wd_in,ht_in),_=cv2.minAreaRect(cnt)
															if ht_in>0 and wd_in>0:
																ratio=(ht/ht_in+wd/wd_in)/2
															else:
																ratio=np.nan
														else:
															area=wd_in=ht_in=ratio=np.nan
														inners_areas[cell_name].append(area)
														inners_heights[cell_name].append(ht_in)
														inners_widths[cell_name].append(wd_in)
														inners_out_ratio[cell_name].append(ratio)

					cv2.imwrite(os.path.join(self.results_path,os.path.splitext(os.path.basename(self.path_to_file))[0]+'_'+str(w)+str(h)+'_annotated.jpg'),to_annotate)

		for cell_name in self.cell_kinds:

			dfs=[]
			if len(cell_centers[cell_name])>0:
				dfs.append(pd.DataFrame([i+1 for i in range(len(cell_centers[cell_name]))],columns=['number']).reset_index(drop=True))
				dfs.append(pd.DataFrame(cell_centers[cell_name],columns=['center_x','center_y']).reset_index(drop=True))
				dfs.append(pd.DataFrame(cell_areas[cell_name],columns=['areas']).reset_index(drop=True))
				dfs.append(pd.DataFrame(cell_heights[cell_name],columns=['heights']).reset_index(drop=True))
				dfs.append(pd.DataFrame(cell_widths[cell_name],columns=['widths']).reset_index(drop=True))
				dfs.append(pd.DataFrame(cell_perimeter[cell_name],columns=['perimeter']).reset_index(drop=True))
				dfs.append(pd.DataFrame(cell_roundness[cell_name],columns=['roundness']).reset_index(drop=True))
				dfs.append(pd.DataFrame(cell_intensities[cell_name],columns=['intensities']).reset_index(drop=True))
				if self.inners is not None:
					dfs.append(pd.DataFrame(inners_areas[cell_name],columns=['inners_areas']).reset_index(drop=True))
					dfs.append(pd.DataFrame(inners_heights[cell_name],columns=['inners_heights']).reset_index(drop=True))
					dfs.append(pd.DataFrame(inners_widths[cell_name],columns=['inners_widths']).reset_index(drop=True))
					dfs.append(pd.DataFrame(inners_out_ratio[cell_name],columns=['inneroutter_ratio']).reset_index(drop=True))
			else:
				dfs.append(pd.DataFrame(['NA'],columns=['number']).reset_index(drop=True))
				dfs.append(pd.DataFrame([('NA','NA')],columns=['center_x','center_y']).reset_index(drop=True))
				dfs.append(pd.DataFrame(['NA'],columns=['areas']).reset_index(drop=True))
				dfs.append(pd.DataFrame(['NA'],columns=['heights']).reset_index(drop=True))
				dfs.append(pd.DataFrame(['NA'],columns=['widths']).reset_index(drop=True))
				dfs.append(pd.DataFrame(['NA'],columns=['perimeter']).reset_index(drop=True))
				dfs.append(pd.DataFrame(['NA'],columns=['roundness']).reset_index(drop=True))
				dfs.append(pd.DataFrame(['NA'],columns=['intensities']).reset_index(drop=True))
				if self.inners is not None:
					dfs.append(pd.DataFrame(['NA'],columns=['inners_areas']).reset_index(drop=True))
					dfs.append(pd.DataFrame(['NA'],columns=['inners_heights']).reset_index(drop=True))
					dfs.append(pd.DataFrame(['NA'],columns=['inners_widths']).reset_index(drop=True))
					dfs.append(pd.DataFrame(['NA'],columns=['inneroutter_ratio']).reset_index(drop=True))
			out_sheet=os.path.join(self.results_path,os.path.splitext(os.path.basename(self.path_to_file))[0]+'_'+cell_name+'_summary.xlsx')
			pd.concat(dfs,axis=1).to_excel(out_sheet,float_format='%.2f',index_label='ID/parameter')

			dfs={}
			dfs['total_area']=total_foreground_area
			dfs[cell_name+'_area']=total_cell_area[cell_name]
			dfs['area_ratio']=total_cell_area[cell_name]/total_foreground_area
			dfs=pd.DataFrame(dfs,index=['value'])
			out_sheet=os.path.join(self.results_path,os.path.splitext(os.path.basename(self.path_to_file))[0]+'_'+cell_name+'_arearatio.xlsx')
			dfs.to_excel(out_sheet,float_format='%.6f')

		print('Analysis completed!')


