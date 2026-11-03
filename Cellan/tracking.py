"""Associate two object types within one image by centroid distance or mask IoU."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def validate_tracking(classes, available_classes, max_distance=None):

	if classes is None:
		return None,None
	if not isinstance(classes,(tuple,list)) or len(classes)!=2 or classes[0]==classes[1] or not all(isinstance(kind,str) and kind for kind in classes):
		raise ValueError('Choose two different object types for ID tracking.')
	if any(kind not in available_classes for kind in classes):
		raise ValueError('Both tracked object types must be selected for analysis.')
	if max_distance is not None:
		max_distance=float(max_distance)
		if not np.isfinite(max_distance) or max_distance<0:
			raise ValueError('Maximum tracking distance must be finite and nonnegative, or None for no limit.')
	return tuple(classes),max_distance


def validate_tracking_method(method='distance',min_iou=0.1):

	if method not in ('distance','iou'):
		raise ValueError('Tracking method must be distance or iou.')
	try:
		min_iou=float(min_iou)
	except (TypeError,ValueError):
		raise ValueError('Minimum mask IoU must be a finite ratio from 0 to 1.') from None
	if not np.isfinite(min_iou) or not 0<=min_iou<=1:
		raise ValueError('Minimum mask IoU must be a finite ratio from 0 to 1.')
	return method,min_iou


def _assign_pairs(values,valid,maximize=False):

	"""Maximize valid-pair count, then optimize their total score."""
	if not valid.any():
		return []
	# Allowed costs are at most 1. Each forbidden pair costs more than the sum
	# of all allowed costs, preserving cardinality before optimizing the score.
	cost=np.full(values.shape,min(values.shape)+1.0)
	if maximize:
		cost[valid]=1-values[valid]
	else:
		cost[valid]=values[valid]/max(float(values[valid].max()),1.0)
	rows,columns=linear_sum_assignment(cost)
	return [(int(i),int(j),float(values[i,j])) for i,j in zip(rows,columns) if valid[i,j]]


def match_centroids(reference_centers,target_centers,max_distance=None):

	"""Return (reference index, target index, distance) for one-to-one pairs.

	Maximize the number of pairs within max_distance, then minimize their total
	Euclidean distance (Hungarian assignment). Each detection is used at most
	once. None means unlimited distance; zero permits coincident centroids only.
	Distances use image pixels, and input indices are zero based.
	"""
	_,max_distance=validate_tracking(('reference','target'),('reference','target'),max_distance)
	centers=[]
	for values in (reference_centers,target_centers):
		array=np.asarray(values,dtype=np.float64)
		if array.shape==(0,):
			array=array.reshape(0,2)
		if array.ndim!=2 or array.shape[1]!=2 or not np.isfinite(array).all():
			raise ValueError('Centroids must be finite (x, y) coordinates with shape (N, 2).')
		centers.append(array)
	if not len(centers[0]) or not len(centers[1]):
		return []
	distances=cdist(centers[0],centers[1])
	valid=np.isfinite(distances)
	if max_distance is not None:
		valid&=distances<=max_distance
	return _assign_pairs(distances,valid)


def _mask_geometry(masks,origins):

	if origins is None:
		origins=[(0,0)]*len(masks)
	if len(origins)!=len(masks):
		raise ValueError('Provide one (x, y) origin for each mask.')
	geometry=[]
	for mask,origin in zip(masks,origins):
		mask=np.asarray(mask)
		if mask.ndim!=2 or not np.isfinite(mask).all():
			raise ValueError('IoU tracking requires finite 2D binary masks.')
		origin=np.asarray(origin,dtype=np.float64)
		if origin.shape!=(2,) or not np.isfinite(origin).all() or not np.equal(origin,np.floor(origin)).all():
			raise ValueError('Mask origins must be integer (x, y) pixel coordinates.')
		x,y=(int(value) for value in origin)
		mask=mask!=0
		geometry.append((mask,x,y,mask.shape[1],mask.shape[0],int(mask.sum())))
	return geometry


def mask_iou_matrix(reference_masks,target_masks,reference_origins=None,target_origins=None):

	"""Compute true pixel IoU, aligning cropped masks in full-image coordinates.

	Each origin is the integer (x, y) position of that mask array's top-left pixel.
	Omit origins when the arrays already use the same canvas. Holes and separate
	components are preserved. Empty masks have IoU zero, including two empties.
	"""
	references=_mask_geometry(reference_masks,reference_origins)
	targets=_mask_geometry(target_masks,target_origins)
	ious=np.zeros((len(references),len(targets)),dtype=np.float64)
	if not targets:
		return ious
	boxes=np.array([(x,y,x+width,y+height) for _,x,y,width,height,_ in targets])
	for i,(reference,rx,ry,rw,rh,reference_area) in enumerate(references):
		if not reference_area:
			continue
		candidates=np.flatnonzero((boxes[:,0]<rx+rw)&(boxes[:,2]>rx)&(boxes[:,1]<ry+rh)&(boxes[:,3]>ry))
		for j in candidates:
			target,tx,ty,tw,th,target_area=targets[j]
			if not target_area:
				continue
			x0,y0=max(rx,tx),max(ry,ty)
			x1,y1=min(rx+rw,tx+tw),min(ry+rh,ty+th)
			intersection=np.count_nonzero(reference[y0-ry:y1-ry,x0-rx:x1-rx]&target[y0-ty:y1-ty,x0-tx:x1-tx])
			ious[i,j]=intersection/(reference_area+target_area-intersection)
	return ious


def match_iou(reference_masks,target_masks,min_iou=0.1,reference_origins=None,target_origins=None):

	"""Return (reference index, target index, IoU) for one-to-one mask matches.

	Maximize the number of pairs meeting min_iou, then their total IoU. Disjoint
	or empty masks never match, even when min_iou is zero. Origins permit cropped
	masks from different positions to be compared in full-image coordinates.
	"""
	_,min_iou=validate_tracking_method('iou',min_iou)
	ious=mask_iou_matrix(reference_masks,target_masks,reference_origins,target_origins)
	return _assign_pairs(ious,(ious>0)&(ious>=min_iou),maximize=True)


def assign_cell_ids(detections,tracking_classes=None,max_distance=None,method='distance',min_iou=0.1):

	"""Return retained detection records with IDs, plus a match audit table.

	Records contain kind and centroid. IoU mode also needs tracking_mask and its
	mask_origin (a tight crop of the original, unexpanded mask in image pixels).
	When tracking is enabled, matched records
	share a 1-based cell_id; unmatched records of either selected kind are dropped.
	Other kinds remain independent and get IDs above the matched range. With
	tracking disabled, preserve independent 1-based IDs within each object type.
	Inputs are copied, so assigning IDs never changes the caller's records.
	"""
	method,min_iou=validate_tracking_method(method,min_iou)
	records=[dict(record,detection_id=index+1) for index,record in enumerate(detections)]
	for record in records:
		record.pop('cell_id',None)
	if tracking_classes is None:
		counts={}
		for record in records:
			kind=record['kind']
			counts[kind]=counts.get(kind,0)+1
			record['cell_id']=counts[kind]
		return records,[]
	tracking_classes,max_distance=validate_tracking(tracking_classes,tracking_classes,max_distance)
	reference_kind,target_kind=tracking_classes
	references=[record for record in records if record['kind']==reference_kind]
	targets=[record for record in records if record['kind']==target_kind]
	if method=='iou':
		if any('tracking_mask' not in record or 'mask_origin' not in record for record in references+targets):
			raise ValueError('IoU association requires each original mask (tracking_mask) and its mask_origin.')
		pairs=match_iou([record['tracking_mask'] for record in references],
			[record['tracking_mask'] for record in targets],min_iou,
			[record['mask_origin'] for record in references],[record['mask_origin'] for record in targets])
	else:
		pairs=match_centroids([record['centroid'] for record in references],
			[record['centroid'] for record in targets],max_distance)
	matches=[]
	for cell_id,(i,j,score) in enumerate(pairs,start=1):
		reference,target=references[i],targets[j]
		reference['cell_id']=target['cell_id']=cell_id
		separation=float(np.linalg.norm(np.subtract(reference['centroid'],target['centroid'])))
		matches.append({'cell_id':cell_id,'reference_class':reference_kind,
			'reference_detection_id':reference['detection_id'],
			'reference_center_x':reference['centroid'][0],'reference_center_y':reference['centroid'][1],
			'target_class':target_kind,'target_detection_id':target['detection_id'],
			'target_center_x':target['centroid'][0],'target_center_y':target['centroid'][1],
			'centroid_distance':separation,'tracking_method':method,'mask_iou':score if method=='iou' else np.nan})
	independent_id=len(pairs)
	for record in records:
		if record['kind'] not in tracking_classes:
			independent_id+=1
			record['cell_id']=independent_id
	return [record for record in records if 'cell_id' in record],matches
