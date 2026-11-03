from .segmentation import prepare_segmentation_jobs, validate_boundary_options, analysis_kinds, filter_inside_boundaries, clear_previous_assignment_outputs, PALETTE
from .tracking import assign_cell_ids, validate_tracking, validate_tracking_method
from .channels import read_channels, validate_channel_indices, validate_mapping, detection_tile
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

	def __init__(self,path_to_file,results_path,path_to_detector=None,cell_kinds=None,names_colors=None,detection_threshold=None,expansion=None,show_ids=False,filters=None,tracking_classes=None,tracking_max_distance=None,tracking_method='distance',tracking_min_iou=0.1,segmentation_jobs=None,boundary_sources=None,boundary_mode='all',boundary_min_fraction=1.0):

		self.path_to_file=path_to_file
		self.results_path=os.path.join(results_path,os.path.splitext(os.path.basename(path_to_file))[0])
		self.lif=os.path.splitext(path_to_file)[1].lower()=='.lif'
		os.makedirs(self.results_path,exist_ok=True)
		self.segmentation_jobs=None
		self.boundary_sources=()
		self.boundary_mode=boundary_mode
		self.boundary_min_fraction=boundary_min_fraction
		self.boundary_detections=[]
		self.boundary_audit=[]
		if segmentation_jobs is not None:
			self.segmentation_jobs=prepare_segmentation_jobs(segmentation_jobs)
			self.boundary_sources,self.boundary_mode,self.boundary_min_fraction=validate_boundary_options(
				boundary_sources,self.segmentation_jobs,boundary_mode,boundary_min_fraction)
			available=analysis_kinds(self.segmentation_jobs,self.boundary_sources)
			self.cell_kinds=list(cell_kinds) if cell_kinds else available
			if any(kind not in available for kind in self.cell_kinds):
				raise ValueError('Analysis object types must belong to non-boundary Detector assignments.')
			all_kinds=[label for job in self.segmentation_jobs for label in job['qualified_kinds'].values()]
			self.names_colors={kind:PALETTE[index%len(PALETTE)] for index,kind in enumerate(all_kinds)}
			if names_colors:
				self.names_colors.update(names_colors)
			self.detection_threshold={label:job['score_threshold'] for job in self.segmentation_jobs for label in job['qualified_kinds'].values()}
			if detection_threshold:
				self.detection_threshold.update(detection_threshold)
			first=next(job for job in self.segmentation_jobs if job['name'] not in self.boundary_sources)
			self.fov_dim=first['inferencing_framesize']
			self.black_background=first['black_background']
			self.detector=None
			self.cell_mapping={}
		else:
			if boundary_sources:
				raise ValueError('Boundary channels require Detector/channel assignments.')
			self.cell_kinds=list(cell_kinds or [])
			if path_to_detector is None:
				raise ValueError('Select a Detector or configure Detector/channel assignments.')
			self.detector=Detector()
			self.detector.load(path_to_detector,self.cell_kinds)
			self.cell_mapping=self.detector.cell_mapping
			self.names_colors=names_colors
			self.detection_threshold=detection_threshold if detection_threshold is not None else {kind:0 for kind in self.cell_kinds}
			self.fov_dim=self.detector.inferencing_framesize
			self.black_background=self.detector.black_background
		self.tracking_classes,self.tracking_max_distance=validate_tracking(tracking_classes,self.cell_kinds,tracking_max_distance)
		self.tracking_method,self.tracking_min_iou=validate_tracking_method(tracking_method,tracking_min_iou)
		self.expansion=expansion
		self.show_ids=show_ids
		self.filters=filters or {}
		self.detections=[]
		self.tracking_matches=[]


	def _prepare_tile_detections(self,masks,classes,scores,valid_shape,offset,apply_filters=False,selected_kinds=None,preserve_raw=False,process_masks=True):

		"""Filter masks and preserve original geometry before optional expansion."""
		height,width=valid_shape
		x0,y0=offset
		records=[]
		for kind in (self.cell_kinds if selected_kinds is None else selected_kinds):
			cell_masks=[]
			for mask,label,score in zip(masks,classes,scores):
				if label!=kind or score<self.detection_threshold.get(kind,0):
					continue
				mask=(mask!=0).astype(np.uint8)
				mask[height:,:]=0
				mask[:,width:]=0
				if mask.any():
					cell_masks.append(mask)
			if not cell_masks:
				continue
			# Remove smaller masks largely contained in another mask of this type.
			areas=np.sum(cell_masks,axis=(1,2))
			overlaps=np.sum(np.logical_and(np.array(cell_masks)[:,None],cell_masks),axis=(2,3))
			excluded=np.any((overlaps/areas[:,None]>0.8)&(areas[:,None]<areas[None,:]),axis=1)
			for mask,exclude in zip(cell_masks,excluded):
				if exclude:
					continue
				y_pixels,x_pixels=np.nonzero(mask)
				centroid=(float(x_pixels.mean())+x0,float(y_pixels.mean())+y0)
				tracking_data={}
				if preserve_raw or (self.tracking_classes is not None and self.tracking_method=='iou' and kind in self.tracking_classes):
					left,top,right,bottom=int(x_pixels.min()),int(y_pixels.min()),int(x_pixels.max())+1,int(y_pixels.max())+1
					tracking_data={'tracking_mask':mask[top:bottom,left:right].astype(bool,copy=True),'mask_origin':(x0+left,y0+top)}
				if process_masks:
					mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
					if self.expansion is not None:
						mask=cv2.dilate(mask,np.ones((5,5),np.uint8),iterations=self.expansion)
				mask[height:,:]=0
				mask[:,width:]=0
				contours,_=cv2.findContours(mask*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
				if not contours:
					continue
				contour=max(contours,key=cv2.contourArea)
				area=int(mask.sum())
				perimeter=float(cv2.arcLength(contour,closed=True))
				(_,_),(wd,ht),_=cv2.minAreaRect(contour)
				metrics={'area':area,'perimeter':perimeter,'height':ht,'width':wd,
					'roundness':4*np.pi*area/(perimeter*perimeter) if perimeter>0 else 0.0}
				if apply_filters and any(not self.filters[key][0]<=value<=self.filters[key][1]
					for key,value in metrics.items() if key in self.filters):
					continue
				records.append({'kind':kind,'centroid':centroid,'center':tuple(int(value) for value in centroid),
					'contour':contour+np.array([[[x0,y0]]]),'mask':mask,'metrics':metrics,
					'tile':(x0,y0,x0+width,y0+height),**tracking_data})
		return records


	def _assign_ids(self,records):

		# Match over the whole image, not separately within each inference tile.
		self.detections,self.tracking_matches=assign_cell_ids(records,self.tracking_classes,self.tracking_max_distance,
			method=self.tracking_method,min_iou=self.tracking_min_iou)
		match_path=os.path.join(self.results_path,'cell_matches.csv')
		if self.tracking_classes is not None:
			columns=['cell_id','reference_class','reference_detection_id','reference_center_x','reference_center_y',
				'target_class','target_detection_id','target_center_x','target_center_y','centroid_distance','tracking_method','mask_iou']
			pd.DataFrame(self.tracking_matches,columns=columns).to_csv(match_path,index=False)
			print('Matched '+str(len(self.tracking_matches))+' cell pairs; discarded '+str(len(records)-len(self.detections))+' unmatched detections.')
		elif os.path.isfile(match_path):
			os.remove(match_path)
		return self.detections


	def _draw_detection(self,image,record,thickness,offset=(0,0)):

		hex_color=self.names_colors[record['kind']].lstrip('#')
		color=tuple(int(hex_color[i:i+2],16) for i in (4,2,0))
		x0,y0=offset
		contour=record['contour']-np.array([[[x0,y0]]])
		cv2.drawContours(image,[contour],0,color,thickness)
		if self.show_ids:
			center=(record['center'][0]-x0,record['center'][1]-y0)
			cv2.putText(image,str(record['cell_id']),center,cv2.FONT_HERSHEY_SIMPLEX,thickness,color,thickness)


	def analyze_multichannels(self,detection_channel=0,analysis_channels=None,downsize_factor=None):

		"""Segment saved composites or a single channel, then optionally match IDs.

		Detector.load() reads channel_color_mapping from model_parameters.txt.
		When present, that mapping selects all detection channels and RGB LUTs;
		detection_channel is used only by detectors without a saved mapping.
		Tracking uses original mask centroids or pixel IoU over the entire image.
		Measurement channels retain their original, uncolorized intensities.
		"""
		if self.segmentation_jobs is not None:
			return self._analyze_multiple_detectors(analysis_channels,downsize_factor)
		channels=read_channels(self.path_to_file)
		mapping=getattr(self.detector,'channel_color_mapping',None)
		if mapping is not None:
			mapping=validate_mapping(mapping)
			detection_channels=[entry['index'] for entry in mapping['channels']]
			print('Reconstructing detection image from saved channel-color mapping: '+str(mapping['channels']))
		else:
			detection_channels=[detection_channel]
			print('Reconstructing detection image from single channel: '+str(detection_channel))
		analysis_channels=list(analysis_channels) if analysis_channels else list(range(channels.shape[0]))
		validate_channel_indices(channels,analysis_channels)
		validate_channel_indices(channels,detection_channels)
		if downsize_factor:
			if downsize_factor<=0:
				raise ValueError('Downsize factor must be positive.')
			new_size=(max(1,int(channels.shape[2]*downsize_factor/100)),max(1,int(channels.shape[1]*downsize_factor/100)))
			channels=np.stack([cv2.resize(plane,new_size,interpolation=cv2.INTER_AREA) for plane in channels])
		if mapping is None:
			detect_image=cv2.cvtColor(np.uint8(channels[detection_channel]),cv2.COLOR_GRAY2BGR)
		height,width=channels.shape[1:]
		records=[]
		tiles=[]
		stem=os.path.splitext(os.path.basename(self.path_to_file))[0]
		for y0 in range(0,height,self.fov_dim):
			for x0 in range(0,width,self.fov_dim):
				y1,x1=min(y0+self.fov_dim,height),min(x0+self.fov_dim,width)
				tiles.append((x0,y0,x1,y1))
				if mapping is not None:
					# Same LUT overlay, normalization, padding and BGR order as training.
					detect_fov=detection_tile(channels[:,y0:y1,x0:x1],mapping,self.fov_dim,self.black_background,bgr=True)
				else:
					fov=np.uint8(exposure.rescale_intensity(detect_image[y0:y1,x0:x1],out_range=(0,255)))
					detect_fov=np.full((self.fov_dim,self.fov_dim,3),0 if self.black_background else 255,dtype=np.uint8)
					detect_fov[:y1-y0,:x1-x0]=fov
				output=self.detector.inference([{'image':torch.as_tensor(detect_fov.astype('float32').transpose(2,0,1))}])
				instances=output[0]['instances'].to('cpu')
				masks=instances.pred_masks.numpy().astype(np.uint8)
				classes=[self.cell_mapping[str(value)] for value in instances.pred_classes.numpy()]
				tile_records=self._prepare_tile_detections(masks,classes,instances.scores.numpy(),(y1-y0,x1-x0),(x0,y0),apply_filters=True)
				for record in tile_records:
					mask=record.pop('mask')[:y1-y0,:x1-x0].astype(bool)
					record['intensities']={c:float(np.sum(channels[c,y0:y1,x0:x1],dtype=np.float64,where=mask)/record['metrics']['area']) for c in analysis_channels}
				records.extend(tile_records)

		records=self._assign_ids(records)
		by_tile={tile:[] for tile in tiles}
		for record in records:
			by_tile[record['tile']].append(record)
		thickness=max(1,round(self.fov_dim/960))
		# Render only after matching, so discarded masks never appear in annotations.
		for tile in tiles:
			x0,y0,x1,y1=tile
			for c in analysis_channels:
				fov=np.zeros((self.fov_dim,self.fov_dim),dtype=channels.dtype)
				fov[:y1-y0,:x1-x0]=channels[c,y0:y1,x0:x1]
				annotated=cv2.cvtColor(np.uint8(exposure.rescale_intensity(fov,out_range=(0,255))),cv2.COLOR_GRAY2BGR)
				for record in by_tile[tile]:
					self._draw_detection(annotated,record,thickness,(x0,y0))
				filename=stem+'_'+str(x0//self.fov_dim)+str(y0//self.fov_dim)+'_c'+str(c)+'_annotated.jpg'
				cv2.imwrite(os.path.join(self.results_path,filename),annotated)

		for kind in self.cell_kinds:
			rows=[]
			for record in sorted((record for record in records if record['kind']==kind),key=lambda record:record['cell_id']):
				rows.append([record['cell_id'],*record['center'],record['metrics']['area']]+[record['intensities'][c] for c in analysis_channels])
			columns=['number','center_x','center_y','areas']+['intensity_'+str(c) for c in analysis_channels]
			pd.DataFrame(rows,columns=columns).to_excel(os.path.join(self.results_path,stem+'_'+kind+'_summary.xlsx'),float_format='%.2f',index_label='ID/parameter')
		print('Analysis completed!')


	def _analyze_multiple_detectors(self,analysis_channels=None,downsize_factor=None):

		channels=read_channels(self.path_to_file)
		analysis_channels=list(analysis_channels) if analysis_channels else list(range(channels.shape[0]))
		validate_channel_indices(channels,analysis_channels)
		# Check every source before loading any model or writing result tables.
		for job in self.segmentation_jobs:
			mapping=job['channel_color_mapping']
			validate_channel_indices(channels,[entry['index'] for entry in mapping['channels']] if mapping is not None else [job['detection_channel']])
		if downsize_factor:
			if downsize_factor<=0:
				raise ValueError('Downsize factor must be positive.')
			size=(max(1,int(channels.shape[2]*downsize_factor/100)),max(1,int(channels.shape[1]*downsize_factor/100)))
			channels=np.stack([cv2.resize(plane,size,interpolation=cv2.INTER_AREA) for plane in channels])
		height,width=channels.shape[1:]
		records=[]
		boundaries=[]
		for job in self.segmentation_jobs:
			print('Segmenting assignment '+job['name']+' using '+job['path_to_detector'])
			detector=Detector()
			detector.load(job['path_to_detector'],job['cell_kinds'])
			frame=detector.inferencing_framesize
			# Assignment remapping changes only source indices, preserving trained LUTs.
			mapping=job['channel_color_mapping']
			is_boundary=job['name'] in self.boundary_sources
			selected=list(job['qualified_kinds'].values())
			if not is_boundary:
				selected=[kind for kind in selected if kind in self.cell_kinds]
			parent_count=0
			for y0 in range(0,height,frame):
				for x0 in range(0,width,frame):
					y1,x1=min(y0+frame,height),min(x0+frame,width)
					if mapping is not None:
						detect_fov=detection_tile(channels[:,y0:y1,x0:x1],mapping,frame,detector.black_background,bgr=True)
					else:
						plane=channels[job['detection_channel'],y0:y1,x0:x1]
						fov=cv2.cvtColor(np.uint8(exposure.rescale_intensity(plane,out_range=(0,255))),cv2.COLOR_GRAY2BGR)
						detect_fov=np.full((frame,frame,3),0 if detector.black_background else 255,dtype=np.uint8)
						detect_fov[:y1-y0,:x1-x0]=fov
					output=detector.inference([{'image':torch.as_tensor(detect_fov.astype('float32').transpose(2,0,1))}])
					instances=output[0]['instances'].to('cpu')
					masks=instances.pred_masks.numpy().astype(np.uint8)
					classes=[job['qualified_kinds'].get(detector.cell_mapping[str(value)]) for value in instances.pred_classes.numpy()]
					tile_records=self._prepare_tile_detections(masks,classes,instances.scores.numpy(),(y1-y0,x1-x0),(x0,y0),
						apply_filters=not is_boundary,selected_kinds=selected,preserve_raw=True,process_masks=not is_boundary)
					for record in tile_records:
						record['source']=job['name']
						mask=record.pop('mask')[:y1-y0,:x1-x0].astype(bool)
						if is_boundary:
							parent_count+=1
							record['boundary_id']=parent_count
							boundaries.append(record)
						else:
							# Containment tests the actual quantified region, including expansion.
							y_pixels,x_pixels=np.nonzero(mask)
							left,top,right,bottom=int(x_pixels.min()),int(y_pixels.min()),int(x_pixels.max())+1,int(y_pixels.max())+1
							record['quant_mask']=mask[top:bottom,left:right].copy()
							record['quant_origin']=(x0+left,y0+top)
							record['candidate_id']=len(records)+1
							record['intensities']={c:float(np.sum(channels[c,y0:y1,x0:x1],dtype=np.float64,where=mask)/record['metrics']['area']) for c in analysis_channels}
							records.append(record)
			# Only one assignment's model is kept resident at a time.
			del detector

		self.boundary_detections=boundaries
		records,self.boundary_audit=filter_inside_boundaries(records,boundaries,self.boundary_sources,self.boundary_mode,self.boundary_min_fraction)
		records=self._assign_ids(records)
		retained_ids={record['candidate_id']:record['cell_id'] for record in records}
		for row in self.boundary_audit:
			row['quantified']=row['candidate_id'] in retained_ids
			row['cell_id']=retained_ids.get(row['candidate_id'])
		stem=os.path.splitext(os.path.basename(self.path_to_file))[0]
		clear_previous_assignment_outputs(self.results_path,stem)
		audit_path=os.path.join(self.results_path,'boundary_filter.csv')
		if self.boundary_sources:
			columns=['candidate_id','analysis_source','analysis_kind','boundary_source','boundary_mask_id','inside_fraction',
				'passes_boundary','passes_boundary_rule','quantified','cell_id']
			pd.DataFrame(self.boundary_audit,columns=columns).to_csv(audit_path,index=False)
		elif os.path.isfile(audit_path):
			os.remove(audit_path)
		with open(os.path.join(self.results_path,'segmentation_assignments.json'),'w',encoding='utf-8') as stream:
			json.dump({'assignments':self.segmentation_jobs,'boundary_sources':self.boundary_sources,
				'boundary_mode':self.boundary_mode,'boundary_min_fraction':self.boundary_min_fraction,
				'measurement_channels':analysis_channels,'downsize_factor':downsize_factor},stream,indent=2)
		for kind in self.cell_kinds:
			rows=[]
			for record in sorted((record for record in records if record['kind']==kind),key=lambda record:record['cell_id']):
				rows.append([record['cell_id'],*record['center'],record['metrics']['area']]+[record['intensities'][c] for c in analysis_channels])
			columns=['number','center_x','center_y','areas']+['intensity_'+str(c) for c in analysis_channels]
			pd.DataFrame(rows,columns=columns).to_excel(os.path.join(self.results_path,stem+'_'+kind+'_summary.xlsx'),float_format='%.2f',index_label='ID/parameter')

		# Render all sources on a common output grid despite different model frames.
		by_tile={(x,y):[] for y in range(0,height,self.fov_dim) for x in range(0,width,self.fov_dim)}
		outlines=list(records)+[dict(record,cell_id='B'+str(record['boundary_id'])) for record in boundaries]
		for record in outlines:
			points=record['contour'].reshape(-1,2)
			left,top=points.min(axis=0)
			right,bottom=points.max(axis=0)
			for y in range(max(0,int(top)//self.fov_dim)*self.fov_dim,min(height,int(bottom)+1),self.fov_dim):
				for x in range(max(0,int(left)//self.fov_dim)*self.fov_dim,min(width,int(right)+1),self.fov_dim):
					by_tile[x,y].append(record)
		thickness=max(1,round(self.fov_dim/960))
		for (x0,y0),tile_records in by_tile.items():
			y1,x1=min(y0+self.fov_dim,height),min(x0+self.fov_dim,width)
			for c in analysis_channels:
				plane=np.zeros((self.fov_dim,self.fov_dim),dtype=channels.dtype)
				plane[:y1-y0,:x1-x0]=channels[c,y0:y1,x0:x1]
				annotated=cv2.cvtColor(np.uint8(exposure.rescale_intensity(plane,out_range=(0,255))),cv2.COLOR_GRAY2BGR)
				for record in tile_records:
					self._draw_detection(annotated,record,thickness,(x0,y0))
				filename=stem+'_x'+str(x0)+'_y'+str(y0)+'_c'+str(c)+'_annotated.jpg'
				cv2.imwrite(os.path.join(self.results_path,filename),annotated)
		print('Analysis completed with '+str(len(records))+' quantified objects.')


	def analyze_singlechannel(self):

		if self.segmentation_jobs is not None:
			raise ValueError('Use Analyze Multichannel Images for multiple Detector assignments.')

		image_name=os.path.basename(self.path_to_file)
		stem=os.path.splitext(image_name)[0]
		if os.path.splitext(image_name)[1].lower() in ['.jpg','.png']:
			image=cv2.imread(self.path_to_file)
		else:
			image=imread(self.path_to_file)
			if image.ndim==2:
				image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
			else:
				image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
		height,width=image.shape[:2]
		total_foreground_area=0
		records=[]
		for y0 in range(0,height,self.fov_dim):
			for x0 in range(0,width,self.fov_dim):
				y1,x1=min(y0+self.fov_dim,height),min(x0+self.fov_dim,width)
				analysis_fov=np.full((self.fov_dim,self.fov_dim,3),0 if self.black_background else 255,dtype=image.dtype)
				analysis_fov[:y1-y0,:x1-x0]=image[y0:y1,x0:x1]
				fov=np.uint8(exposure.rescale_intensity(image[y0:y1,x0:x1],out_range=(0,255)))
				detect_fov=np.full((self.fov_dim,self.fov_dim,3),0 if self.black_background else 255,dtype=np.uint8)
				detect_fov[:y1-y0,:x1-x0]=fov
				foreground=cv2.threshold(cv2.cvtColor(fov,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
				total_foreground_area+=np.count_nonzero(foreground) if self.black_background else foreground.size-np.count_nonzero(foreground)
				output=self.detector.inference([{'image':torch.as_tensor(detect_fov.astype('float32').transpose(2,0,1))}])
				instances=output[0]['instances'].to('cpu')
				masks=instances.pred_masks.numpy().astype(np.uint8)
				classes=[self.cell_mapping[str(value)] for value in instances.pred_classes.numpy()]
				tile_records=self._prepare_tile_detections(masks,classes,instances.scores.numpy(),(y1-y0,x1-x0),(x0,y0),apply_filters=True)
				for record in tile_records:
					mask=record.pop('mask').astype(bool)
					record['intensity']=float(np.sum(analysis_fov,dtype=np.float64,where=mask[...,None])/record['metrics']['area'])
				records.extend(tile_records)

		records=self._assign_ids(records)
		annotated=np.uint8(exposure.rescale_intensity(image,out_range=(0,255)))
		thickness=max(1,round(self.fov_dim/960))
		for record in records:
			self._draw_detection(annotated,record,thickness)
		cv2.imwrite(os.path.join(self.results_path,stem+'_annotated.'+image_name.split('.')[-1]),annotated)
		parameters=['center','area','height','width','perimeter','roundness','intensity']
		with pd.ExcelWriter(os.path.join(self.results_path,stem+'_summary.xlsx'),engine='openpyxl') as writer:
			for kind in self.cell_kinds:
				rows=[]
				for record in sorted((record for record in records if record['kind']==kind),key=lambda record:record['cell_id']):
					metrics=record['metrics']
					rows.append([stem,record['cell_id'],record['center']]+[metrics[key] for key in parameters[1:-1]]+[record['intensity']])
				pd.DataFrame(rows,columns=['filename','ID']+parameters).to_excel(writer,sheet_name=kind,float_format='%.2f',index=False)
		with pd.ExcelWriter(os.path.join(self.results_path,stem+'_arearatio.xlsx'),engine='openpyxl') as writer:
			for kind in self.cell_kinds:
				area=sum(record['metrics']['area'] for record in records if record['kind']==kind)
				ratio=area/total_foreground_area if total_foreground_area else np.nan
				pd.DataFrame({'total_area':total_foreground_area,kind+'_area':area,'area_ratio':ratio},index=['value']).to_excel(writer,sheet_name=kind,float_format='%.6f')

		# COCO annotation IDs remain unique; cell_id links the two object types.
		coco_format={'info':{'year':'','version':'1','description':'Cellan annotations','contributor':'','url':'https://github.com/yujiahu415/Cellan','date_created':''},'licenses':[],'categories':[],'images':[],'annotations':[]}
		for index,kind in enumerate(self.cell_kinds):
			coco_format['categories'].append({'id':index+1,'name':kind,'supercategory':'none'})
		if os.path.splitext(image_name)[1].lower() in ['.jpg','.png']:
			cv2.imwrite(os.path.join(self.results_path,image_name),image)
		else:
			imwrite(os.path.join(self.results_path,image_name),cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
		coco_format['images'].append({'id':0,'width':width,'height':height,'file_name':image_name})
		for annotation_id,record in enumerate(records):
			contour=record['contour']
			x,y,wd,ht=cv2.boundingRect(contour)
			coco_format['annotations'].append({'id':annotation_id,'cell_id':record['cell_id'],'image_id':0,
				'category_id':self.cell_kinds.index(record['kind'])+1,'segmentation':[contour.flatten().tolist()],
				'area':float(cv2.contourArea(contour)),'bbox':[x,y,wd-1,ht-1],'iscrowd':0})
		with open(os.path.join(self.results_path,'annotations.json'),'w') as stream:
			json.dump(coco_format,stream)
		print('Analysis completed!')



class AnalyzeCalciumSignal():

	def __init__(self,path_to_lif,results_path,stim_t,duration):

		self.detector=None
		self.cell_mapping=None
		self.path_to_lif=path_to_lif
		self.results_path=os.path.join(results_path,os.path.splitext(os.path.basename(self.path_to_lif))[0])
		if os.path.splitext(os.path.basename(self.path_to_lif))[1] in ['.tif','.TIF','.tiff','.TIFF']:
			self.tif=True
		else:
			self.tif=False
		os.makedirs(self.results_path,exist_ok=True)
		self.cell_number=None
		self.cell_kinds=None  # the catgories of neural structures to be analyzed
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
		self.cell_contours={}
		self.cell_centers={}
		self.cell_existingcenters={}
		self.cell_masks={}
		self.cell_areas={}
		self.cell_Fmeans={}
		self.cell_correctFmeans={}
		self.all_parameters={}


	def prepare_analysis(self,path_to_detector,cell_number,cell_kinds):
		
		print('Preparation started...')
		print(datetime.datetime.now())

		self.detector=Detector()
		self.detector.load(path_to_detector,cell_kinds)
		self.cell_mapping=self.detector.cell_mapping
		self.cell_number=cell_number
		self.cell_kinds=cell_kinds

		total_number=0

		for cell_name in self.cell_kinds:

			total_number+=self.cell_number[cell_name]
			self.all_parameters[cell_name]={}
			for parameter_name in ['F0','dF/F0','Fmax','Stim_t']:
				self.all_parameters[cell_name][parameter_name]={}
			self.to_deregister[cell_name]={}
			self.register_counts[cell_name]={}
			self.cell_contours[cell_name]={}
			self.cell_centers[cell_name]={}
			self.cell_existingcenters[cell_name]={}
			self.cell_masks[cell_name]={}
			self.cell_areas[cell_name]={}
			self.cell_Fmeans[cell_name]={}
			self.cell_correctFmeans[cell_name]={}

			for i in range(self.cell_number[cell_name]):
				self.to_deregister[cell_name][i]=0
				self.register_counts[cell_name][i]=None
				self.cell_contours[cell_name][i]=[None]*self.duration
				self.cell_centers[cell_name][i]=[None]*self.duration
				self.cell_existingcenters[cell_name][i]=(-10000,-10000)
				self.cell_masks[cell_name][i]=[None]*self.duration
				self.cell_areas[cell_name][i]=[None]*self.duration
				self.cell_Fmeans[cell_name][i]=[0.0]*self.duration
				self.cell_correctFmeans[cell_name][i]=[0.0]*self.duration

		print('Preparation completed!')


	def track_cell(self,frame_count,cell_name,contours,centers,masks,areas,Fmeans):

		unused_existing_indices=list(self.cell_existingcenters[cell_name])
		existing_centers=list(self.cell_existingcenters[cell_name].values())
		unused_new_indices=list(range(len(centers)))
		dt_flattened=distance.cdist(existing_centers,centers).flatten()
		dt_sort_index=dt_flattened.argsort()
		length=len(centers)

		for idx in dt_sort_index:
			index_in_existing=int(idx/length)
			index_in_new=int(idx%length)
			if self.cell_existingcenters[cell_name][index_in_existing][0]==-10000:
				dt=np.inf
			else:
				dt=50
			if dt_flattened[idx]<dt:
				if index_in_existing in unused_existing_indices:
					if index_in_new in unused_new_indices:
						unused_existing_indices.remove(index_in_existing)
						unused_new_indices.remove(index_in_new)
						if self.register_counts[cell_name][index_in_existing] is None:
							self.register_counts[cell_name][index_in_existing]=frame_count
						self.to_deregister[cell_name][index_in_existing]=0
						self.cell_contours[cell_name][index_in_existing][frame_count]=contours[index_in_new]
						center=centers[index_in_new]
						self.cell_centers[cell_name][index_in_existing][frame_count]=center
						self.cell_existingcenters[cell_name][index_in_existing]=center
						self.cell_masks[cell_name][index_in_existing][frame_count]=masks[index_in_new]
						self.cell_areas[cell_name][index_in_existing][frame_count]=areas[index_in_new]
						self.cell_Fmeans[cell_name][index_in_existing][frame_count]=Fmeans[index_in_new]

		'''
		if len(unused_existing_indices)>0:
			for i in unused_existing_indices:
				if self.to_deregister[cell_name][i]<5:
					self.to_deregister[cell_name][i]+=1
				else:
					self.cell_existingcenters[cell_name][i]=(-10000,-10000)
		'''


	def detect_cell(self,frames,images,batch_size,frame_count):

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
				classes=[self.cell_mapping[str(x)] for x in classes]
				scores=[s for s,exclude in zip(scores,exclusion_mask) if not exclude]

				for cell_name in self.cell_kinds:

					contours=[]
					centers=[]
					goodcontours=[]
					goodmasks=[]
					final_masks=[]
					final_areas=[]
					Fmeans=[]

					cell_number=int(self.cell_number[cell_name])
					cell_masks=[masks[a] for a,name in enumerate(classes) if name==cell_name]
					cell_scores=[scores[a] for a,name in enumerate(classes) if name==cell_name]

					if len(cell_masks)>0:

						if len(cell_scores)>cell_number*2:
							sorted_scores_indices=np.argsort(cell_scores)[-int(cell_number*2):]
							cell_masks=[cell_masks[x] for x in sorted_scores_indices]

						for mask in cell_masks:
							mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
							goodmasks.append(mask)
							cnts,_=cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
							goodcontours.append(sorted(cnts,key=cv2.contourArea,reverse=True)[0])

						areas=[np.sum(np.array(m),axis=(0,1)) for m in goodmasks]
						sorted_area_indices=np.argsort(np.array(areas))[-cell_number:]
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

						self.track_cell(frame_count+1-batch_size+batch_count,cell_name,contours,centers,final_masks,final_areas,Fmeans)


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
				frame_project=exposure.rescale_intensity(frame_project,out_range=(0,255))
				frame_project=cv2.cvtColor(np.uint8(frame_project),cv2.COLOR_GRAY2BGR)
				images.append(frame_project)

				batch_count+=1

				if batch_count==batch_size:
					batch_count=0
					self.detect_cell(main_frames,images,batch_size,frame_count)
					main_frames=[]
					images=[]

				if (frame_count+1)%10==0:
					print(str(frame_count+1)+' frames processed...')
					print(datetime.datetime.now())

			if frame_count>=self.full_duration:
				if len(main_frames)>0:
					self.detect_cell(main_frames,images,batch_size,frame_count)
				break

			frame_count+=1

		print('Information acquisition completed!')


	def craft_data(self):

		print('Crafting data...')
		print(datetime.datetime.now())

		for cell_name in self.cell_kinds:

			to_delete=[]
			IDs=list(self.cell_centers[cell_name].keys())

			for i in IDs:
				if self.register_counts[cell_name][i] is None:
					to_delete.append(i)

			if len(IDs)==len(to_delete):
				print('No neural structure detected!')
			
			for i in IDs:
				if i in to_delete:
					del self.to_deregister[cell_name][i]
					del self.register_counts[cell_name][i]
					del self.cell_centers[cell_name][i]
					del self.cell_existingcenters[cell_name][i]
					del self.cell_contours[cell_name][i]
					del self.cell_masks[cell_name][i]
					del self.cell_areas[cell_name][i]
					del self.cell_Fmeans[cell_name][i]
					del self.cell_correctFmeans[cell_name][i]

			centers=[]
			contours=[]
			masks=[]
			areas=[]
			Fmeans=[]
			correctFmeans=[]

			for i in self.cell_centers[cell_name]:

				temp_centers=[c for c in self.cell_centers[cell_name][i] if c is not None]
				temp_contours=[c for c in self.cell_contours[cell_name][i] if c is not None]
				temp_masks=[m for m in self.cell_masks[cell_name][i] if m is not None]
				temp_areas=[a for a in self.cell_areas[cell_name][i] if a is not None]

				idx=np.argsort(temp_areas)[-1]

				centers.append(temp_centers[idx])
				contours.append(temp_contours[idx])
				masks.append(temp_masks[idx])
				areas.append(temp_areas[idx])
				Fmeans.append(self.cell_Fmeans[cell_name][i])
				correctFmeans.append(self.cell_correctFmeans[cell_name][i])

			self.cell_centers[cell_name]={}
			self.cell_contours[cell_name]={}
			self.cell_masks[cell_name]={}
			self.cell_areas[cell_name]={}
			self.cell_Fmeans[cell_name]={}
			self.cell_correctFmeans[cell_name]={}

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
				self.cell_centers[cell_name][i]=centers[i]
				self.cell_contours[cell_name][i]=contours[i]
				self.cell_masks[cell_name][i]=masks[i]
				self.cell_areas[cell_name][i]=areas[i]
				self.cell_Fmeans[cell_name][i]=Fmeans[i]
				self.cell_correctFmeans[cell_name][i]=correctFmeans[i]

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

				frame_project=exposure.rescale_intensity(frame_project,out_range=(0,255))
				frame_project=cv2.cvtColor(np.uint8(frame_project),cv2.COLOR_GRAY2BGR)

				if writer is None:
					(h,w)=frame_project.shape[:2]
					out=os.path.join(self.results_path,'Annotated video.avi')
					writer=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'MJPG'),1,(w,h),True)

				for cell_name in self.cell_kinds:
					for i in self.cell_centers[cell_name]:
						cx=self.cell_centers[cell_name][i][0]
						cy=self.cell_centers[cell_name][i][1]
						cv2.putText(frame_project,cell_name+str(i),(cx-1,cy+1),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
						ct=self.cell_contours[cell_name][i]
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

				for cell_name in self.cell_kinds:

					for i in self.cell_centers[cell_name]:

						mask=self.cell_masks[cell_name][i]
						area=self.cell_areas[cell_name][i]

						self.cell_correctFmeans[cell_name][i][frame_count]=np.sum(frame_project*mask)/area

				if (frame_count+1)%10==0:
					print(str(frame_count+1)+' frames quantified...')
					print(datetime.datetime.now())

			if frame_count>=self.full_duration:
				break

			frame_count+=1

		for cell_name in self.cell_kinds:

			for i in self.cell_correctFmeans[cell_name]:

				df=pd.DataFrame(self.cell_correctFmeans[cell_name],index=[i for i in range(self.duration)])
				df.to_excel(os.path.join(self.results_path,cell_name+'_F.xlsx'),float_format='%.2f',index_label='frame/ID')

				if self.stim_t<=F0_period:
					F_array=self.cell_correctFmeans[cell_name][i][:(self.stim_t-1)]
				else:
					F_array=self.cell_correctFmeans[cell_name][i][(self.stim_t-F0_period-1):(self.stim_t-1)]

				F0=np.array(F_array).mean()
				self.all_parameters[cell_name]['F0'][i]=F0

				if self.stim_t+F_period>=self.duration:
					F_array=self.cell_correctFmeans[cell_name][i][self.stim_t:]
				else:
					F_array=self.cell_correctFmeans[cell_name][i][self.stim_t:(self.stim_t+F_period)]

				Fmax=np.array(F_array).max()
				self.all_parameters[cell_name]['Fmax'][i]=Fmax

				if F0==0.0:
					print('The F0 of '+cell_name+' '+str(i)+' is 0.')
					self.all_parameters[cell_name]['dF/F0'][i]=np.nan
				else:
					self.all_parameters[cell_name]['dF/F0'][i]=(Fmax-F0)/F0

				self.all_parameters[cell_name]['Stim_t'][i]=self.stim_t

		parameters=[]

		for parameter_name in ['F0','Fmax','dF/F0','Stim_t']:
			df=self.all_parameters[cell_name][parameter_name]
			parameters.append(pd.DataFrame.from_dict(df,orient='index',columns=[parameter_name]).reset_index(drop=True))

		out_sheet=os.path.join(self.results_path,cell_name+'_summary.xlsx')
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

					frame_project=exposure.rescale_intensity(frame_project,out_range=(0,255))
					out_image=os.path.join(self.results_path,str(channel)+'_'+str(frame_count)+'.jpg')
					cv2.imwrite(out_image,np.uint8(np.array(frame_project)))

		print('The images stored in: '+self.results_path)

