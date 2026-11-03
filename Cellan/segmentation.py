"""Detector/channel assignments and pixel-based boundary containment."""
import json
import os
import re

import numpy as np

from .channels import validate_mapping
from .tracking import _mask_geometry

PALETTE=('#00FF00','#FF0000','#00BFFF','#FF00FF','#FFFF00','#FFA500','#00FFFF','#FFFFFF')


def read_detector_parameters(folder):

	with open(os.path.join(folder,'model_parameters.txt'),encoding='utf-8') as stream:
		parameters=json.load(stream)
	if not isinstance(parameters,dict):
		raise ValueError('Detector parameters must be a JSON object.')
	names=parameters.get('cell_names')
	if not isinstance(names,list) or not names or any(not isinstance(name,str) or not name for name in names):
		raise ValueError('Detector parameters must list cell_names.')
	return parameters


def prepare_segmentation_jobs(jobs):

	if not isinstance(jobs,(list,tuple)) or not jobs:
		raise ValueError('Add at least one Detector/channel assignment.')
	prepared=[]
	seen=set()
	labels=set()
	for job in jobs:
		if not isinstance(job,dict):
			raise ValueError('Each segmentation assignment must be a dictionary.')
		name=job.get('name','')
		if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',name):
			raise ValueError('Assignment names must start with a letter/number and use letters, numbers, underscores or hyphens.')
		if name.casefold() in seen:
			raise ValueError('Each Detector assignment needs a unique name.')
		seen.add(name.casefold())
		folder=job.get('path_to_detector')
		if not isinstance(folder,(str,os.PathLike)) or not str(folder):
			raise ValueError('Select a Detector folder for '+name+'.')
		folder=os.path.abspath(os.fspath(folder))
		parameters=read_detector_parameters(folder)
		kinds=job.get('cell_kinds',parameters['cell_names'])
		if not isinstance(kinds,(list,tuple)) or not kinds or len(set(kinds))!=len(kinds) or any(kind not in parameters['cell_names'] for kind in kinds):
			raise ValueError('Select one or more valid object types for '+name+'.')
		if any(re.search(r'[/\\:*?"<>|]',kind) for kind in kinds):
			raise ValueError('Object type names must be valid filename components.')
		qualified={kind:name+'__'+kind for kind in kinds}
		for label in qualified.values():
			if label.casefold() in labels:
				raise ValueError('Assignment/object names produce duplicate output names.')
			labels.add(label.casefold())
		channel=job.get('detection_channel',0)
		if type(channel) is not int or channel<0:
			raise ValueError('Source channels must be nonnegative integers; the first channel is 0.')
		threshold=float(job.get('score_threshold',0.0))
		if not np.isfinite(threshold) or not 0<=threshold<=1:
			raise ValueError('Detection confidence thresholds must be between 0 and 1.')
		frame=int(parameters['inferencing_framesize'])
		if frame<=0:
			raise ValueError('Detector frame size must be positive.')
		mapping=parameters.get('channel_color_mapping')
		remap=job.get('channel_remap') or {}
		if not isinstance(remap,dict):
			raise ValueError('Channel reassignment must map trained channel indices to source channel indices.')
		remap={str(key):value for key,value in remap.items()}
		if mapping is not None:
			mapping=validate_mapping(mapping)
			trained={str(entry['index']) for entry in mapping['channels']}
			if any(key not in trained or type(value) is not int or value<0 for key,value in remap.items()):
				raise ValueError('Reassign only trained channels to nonnegative integer source channels.')
			mapping=validate_mapping(dict(mapping,channels=[dict(entry,index=remap.get(str(entry['index']),entry['index'])) for entry in mapping['channels']]))
		elif remap:
			raise ValueError('For a single-channel Detector, use detection_channel rather than channel_remap.')
		prepared.append({'name':name,'path_to_detector':folder,'detection_channel':channel,
			'cell_kinds':list(kinds),'score_threshold':threshold,'qualified_kinds':qualified,
			'inferencing_framesize':frame,'black_background':int(parameters['black_background'])==0,
			'channel_color_mapping':mapping,'channel_remap':remap})
	return prepared


def validate_boundary_options(sources,jobs,mode='all',min_fraction=1.0):

	if sources is None:
		sources=[]
	if not isinstance(sources,(list,tuple)) or len(set(sources))!=len(sources):
		raise ValueError('Choose boundary assignments without duplicates.')
	available={job['name'] for job in jobs}
	if any(source not in available for source in sources):
		raise ValueError('Each boundary channel must have a configured Detector assignment.')
	if len(sources)==len(jobs):
		raise ValueError('Keep at least one Detector assignment for analysis rather than boundary filtering.')
	if mode not in ('all','any'):
		raise ValueError('Boundary rule must be all or any.')
	try:
		min_fraction=float(min_fraction)
	except (TypeError,ValueError):
		raise ValueError('Minimum inside fraction must be a number greater than 0 and no greater than 1.') from None
	if not np.isfinite(min_fraction) or not 0<min_fraction<=1:
		raise ValueError('Minimum inside fraction must be greater than 0 and no greater than 1.')
	return tuple(sources),mode,min_fraction


def analysis_kinds(jobs,boundary_sources):

	return [label for job in jobs if job['name'] not in boundary_sources for label in job['qualified_kinds'].values()]


def clear_previous_assignment_outputs(folder,stem):

	"""Remove managed outputs from the preceding assignment run before rewrite."""
	manifest=os.path.join(folder,'segmentation_assignments.json')
	if not os.path.isfile(manifest):
		return
	try:
		with open(manifest,encoding='utf-8') as stream:
			previous=json.load(stream)
		old_kinds=[label for job in previous['assignments'] for label in job['qualified_kinds'].values()]
	except (ValueError,OSError,KeyError,TypeError):
		return
	for kind in old_kinds:
		if isinstance(kind,str) and not re.search(r'[/\\]',kind):
			path=os.path.join(folder,stem+'_'+kind+'_summary.xlsx')
			if os.path.isfile(path):
				os.remove(path)
	pattern=re.compile(re.escape(stem)+r'_x\d+_y\d+_c\d+_annotated\.jpg')
	for filename in os.listdir(folder):
		if pattern.fullmatch(filename):
			os.remove(os.path.join(folder,filename))


def inside_fraction(analysis_mask,analysis_origin,boundary_mask,boundary_origin):

	"""Fraction of the analysis mask's pixels covered by one boundary object."""
	return _inside_fraction(_mask_geometry([analysis_mask],[analysis_origin])[0],
		_mask_geometry([boundary_mask],[boundary_origin])[0])


def _inside_fraction(analysis_geometry,boundary_geometry):

	analysis,ax,ay,aw,ah,area=analysis_geometry
	boundary,bx,by,bw,bh,_=boundary_geometry
	if not area:
		return 0.0
	x0,y0=max(ax,bx),max(ay,by)
	x1,y1=min(ax+aw,bx+bw),min(ay+ah,by+bh)
	if x1<=x0 or y1<=y0:
		return 0.0
	intersection=np.count_nonzero(analysis[y0-ay:y1-ay,x0-ax:x1-ax]&boundary[y0-by:y1-by,x0-bx:x1-bx])
	return intersection/area


def filter_inside_boundaries(records,boundaries,sources,mode='all',min_fraction=1.0):

	"""Keep analysis masks inside one parent object per selected boundary source.

	Full containment is the default. For multiple sources, all requires a valid
	parent in each source, while any requires a valid parent in at least one.
	Boundary objects are not unioned together to manufacture a containing parent.
	"""
	if not sources:
		return list(records),[]
	# Validate the public helper even when called outside AnalyzeCells.
	if mode not in ('all','any') or not np.isfinite(min_fraction) or not 0<min_fraction<=1:
		raise ValueError('Choose all/any and an inside fraction greater than 0 and no greater than 1.')
	parents={source:[] for source in sources}
	for boundary in boundaries:
		if boundary['source'] in parents:
			geometry=_mask_geometry([boundary['tracking_mask']],[boundary['mask_origin']])[0]
			parents[boundary['source']].append((boundary['boundary_id'],geometry))
	retained=[]
	audit=[]
	for index,record in enumerate(records,start=1):
		rows=[]
		analysis_geometry=_mask_geometry([record['quant_mask']],[record['quant_origin']])[0]
		for source in sources:
			best_fraction=0.0
			best_id=None
			for boundary_id,boundary_geometry in parents[source]:
				fraction=_inside_fraction(analysis_geometry,boundary_geometry)
				if fraction>best_fraction:
					best_fraction,best_id=fraction,boundary_id
			rows.append({'candidate_id':record.get('candidate_id',index),'analysis_source':record['source'],
				'analysis_kind':record['kind'],'boundary_source':source,'boundary_mask_id':best_id,
				'inside_fraction':best_fraction,'passes_boundary':best_fraction>=min_fraction})
		passes=all(row['passes_boundary'] for row in rows) if mode=='all' else any(row['passes_boundary'] for row in rows)
		for row in rows:
			row['passes_boundary_rule']=passes
		audit.extend(rows)
		if passes:
			retained.append(record)
	return retained,audit
