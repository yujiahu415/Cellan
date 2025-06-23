import os
import cv2
import numpy as np
import wx
import wx.lib.agw.hyperlink as hl
from pathlib import Path
import matplotlib as mpl
import matplotlib.cm as cm
import json
import shutil
import tifffile
import pandas as pd
from skimage.transform import resize
from .analyzer import AnalyzeCells
from .detector import Detector
from .tools import extract_images,preprocess_image,calculate_totalintensity
from Cellan import __version__



the_absolute_current_path=str(Path(__file__).resolve().parent)



class ColorPicker(wx.Dialog):

	def __init__(self,parent,title,name_and_color):

		super(ColorPicker,self).__init__(parent=None,title=title,size=(200,200))

		self.name_and_color=name_and_color
		name=self.name_and_color[0]
		hex_color=self.name_and_color[1].lstrip('#')
		color=tuple(int(hex_color[i:i+2],16) for i in (0,2,4))

		boxsizer=wx.BoxSizer(wx.VERTICAL)

		self.color_picker=wx.ColourPickerCtrl(self,colour=color)

		button=wx.Button(self,wx.ID_OK,label='Apply')

		boxsizer.Add(0,10,0)
		boxsizer.Add(self.color_picker,0,wx.ALL|wx.CENTER,10)
		boxsizer.Add(button,0,wx.ALL|wx.CENTER,10)
		boxsizer.Add(0,10,0)

		self.SetSizer(boxsizer)



class InitialWindow(wx.Frame):

	def __init__(self,title):

		super(InitialWindow,self).__init__(parent=None,title=title,size=(750,440))
		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		self.text_welcome=wx.StaticText(panel,label='Welcome to Cellan!',style=wx.ALIGN_CENTER|wx.ST_ELLIPSIZE_END)
		boxsizer.Add(0,60,0)
		boxsizer.Add(self.text_welcome,0,wx.LEFT|wx.RIGHT|wx.EXPAND,5)
		boxsizer.Add(0,60,0)
		self.text_developers=wx.StaticText(panel,
			label='\nDeveloped by Yujia Hu\n',
			style=wx.ALIGN_CENTER|wx.ST_ELLIPSIZE_END)
		boxsizer.Add(self.text_developers,0,wx.LEFT|wx.RIGHT|wx.EXPAND,5)
		boxsizer.Add(0,60,0)
		
		links=wx.BoxSizer(wx.HORIZONTAL)
		homepage=hl.HyperLinkCtrl(panel,0,'Home Page',URL='https://github.com/yujiahu415/Cellan')
		userguide=hl.HyperLinkCtrl(panel,0,'Extended Guide',URL='')
		links.Add(homepage,0,wx.LEFT|wx.EXPAND,10)
		links.Add(userguide,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(links,0,wx.ALIGN_CENTER,50)
		boxsizer.Add(0,50,0)

		module_modules=wx.BoxSizer(wx.HORIZONTAL)
		button_preprocess=wx.Button(panel,label='Preprocessing Module',size=(200,40))
		button_preprocess.Bind(wx.EVT_BUTTON,self.window_preprocess)
		wx.Button.SetToolTip(button_preprocess,'Enhance image contrast / crop images to exclude unnecessary region / downsize images to make the analysis more efficient.')
		button_train=wx.Button(panel,label='Training Module',size=(200,40))
		button_train.Bind(wx.EVT_BUTTON,self.window_train)
		wx.Button.SetToolTip(button_train,'Teach Cellan to recognize the cells of your interest.')
		button_analyze=wx.Button(panel,label='Analysis Module',size=(200,40))
		button_analyze.Bind(wx.EVT_BUTTON,self.window_analyze)
		wx.Button.SetToolTip(button_analyze,'Use Cellan to analyze cells in images.')
		module_modules.Add(button_preprocess,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_modules.Add(button_train,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_modules.Add(button_analyze,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_modules,0,wx.ALIGN_CENTER,50)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def window_preprocess(self,event):

		WindowLv1_ProcessModule('Preprocessing Module')


	def window_train(self,event):

		WindowLv1_TrainingModule('Training Module')


	def window_analyze(self,event):

		WindowLv1_AnalysisModule('Analysis Module')



class WindowLv1_ProcessModule(wx.Frame):

	def __init__(self,title):

		super(WindowLv1_ProcessModule,self).__init__(parent=None,title=title,size=(500,230))
		self.display_window()


	def display_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,40,0)

		button_processvideos=wx.Button(panel,label='Preprocess Images',size=(300,40))
		button_processvideos.Bind(wx.EVT_BUTTON,self.process_images)
		wx.Button.SetToolTip(button_processvideos,'Enhance image contrast / crop images to exclude unnecessary region / downsize images to make the analysis more efficient.')
		boxsizer.Add(button_processvideos,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,20,0)

		button_drawmarkers=wx.Button(panel,label='Draw Markers',size=(300,40))
		button_drawmarkers.Bind(wx.EVT_BUTTON,self.draw_markers)
		wx.Button.SetToolTip(button_drawmarkers,'Draw locational markers in images.')
		boxsizer.Add(button_drawmarkers,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,30,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def process_images(self,event):

		WindowLv2_ProcessImages('Preprocess Images')


	def draw_markers(self,event):

		WindowLv2_DrawMarkers('Draw Markers')



class WindowLv2_ProcessImages(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_ProcessImages,self).__init__(parent=None,title=title,size=(1000,350))
		self.path_to_images=None
		self.gray_scale=False
		self.downsize_factor=None
		self.result_path=None
		self.enhance_contrast=False
		self.contrast=1.0
		self.crop_image=False
		self.left=0
		self.right=0
		self.top=0
		self.bottom=0

		self.display_window()


	def display_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_inputimages=wx.BoxSizer(wx.HORIZONTAL)
		button_inputimages=wx.Button(panel,label='Select the image(s)\nfor preprocessing',size=(300,40))
		button_inputimages.Bind(wx.EVT_BUTTON,self.select_images)
		wx.Button.SetToolTip(button_inputimages,'Select one or more images. Supported file formats: lif, jpg, png, tif, svs, qptiff.')
		self.text_inputimages=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_inputimages.Add(button_inputimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_inputimages.Add(self.text_inputimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_inputimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe processed images',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'The preprocessed images will be stored in this folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_downsizeimages=wx.BoxSizer(wx.HORIZONTAL)
		button_downsizeimages=wx.Button(panel,label='Specify whether to\ndownsize the images',size=(300,40))
		button_downsizeimages.Bind(wx.EVT_BUTTON,self.downsize_images)
		wx.Button.SetToolTip(button_downsizeimages,'Downsizing images can increase the processing speed.')
		self.text_downsizeimages=wx.StaticText(panel,label='Default: not to downsize the images.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_downsizeimages.Add(button_downsizeimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_downsizeimages.Add(self.text_downsizeimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_downsizeimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_cropimage=wx.BoxSizer(wx.HORIZONTAL)
		button_cropimage=wx.Button(panel,label='Specify whether to\ncrop the images',size=(300,40))
		button_cropimage.Bind(wx.EVT_BUTTON,self.crop_images)
		wx.Button.SetToolTip(button_cropimage,'Cropping images to exclude unnecessary areas can increase the analysis efficiency. You need to specify the 4 corner points of the cropping window. This cropping window will be applied for all images selected.')
		self.text_cropimage=wx.StaticText(panel,label='Default: not to crop images.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_cropimage.Add(button_cropimage,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_cropimage.Add(self.text_cropimage,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_cropimage,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_enhancecontrast=wx.BoxSizer(wx.HORIZONTAL)
		button_enhancecontrast=wx.Button(panel,label='Specify whether to enhance\nthe contrast in images',size=(300,40))
		button_enhancecontrast.Bind(wx.EVT_BUTTON,self.enhance_contrasts)
		wx.Button.SetToolTip(button_enhancecontrast,'Enhancing image contrast will increase the detection accuracy. Enter a contrast value to see whether it is good to apply or re-enter it.')
		self.text_enhancecontrast=wx.StaticText(panel,label='Default: not to enhance contrast.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_enhancecontrast.Add(button_enhancecontrast,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_enhancecontrast.Add(self.text_enhancecontrast,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_enhancecontrast,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_preprocessimages=wx.Button(panel,label='Start to preprocess the images',size=(300,40))
		button_preprocessimages.Bind(wx.EVT_BUTTON,self.preprocess_images)
		wx.Button.SetToolTip(button_preprocessimages,'Preprocess each selected image.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_preprocessimages,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_images(self,event):

		wildcard='Image files(*.lif;*.jpg;*.png;*.tif;*.tiff;*.svs;*.qptiff)|*.lif;*.LIF;*.jpg;*.JPG;*.png;*.PNG;*.tif;*.TIF;*.tiff;*.TIFF;*.svs;*.SVS;*.qptiff;*.QPTIFF'
		dialog=wx.FileDialog(self,'Select image(s)','','',wildcard,style=wx.FD_MULTIPLE)

		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_images=dialog.GetPaths()
			self.path_to_images.sort()
			path=os.path.dirname(self.path_to_images[0])
			dialog1=wx.MessageDialog(self,'Turn images to gray scale?','Gray scale?',wx.YES_NO|wx.ICON_QUESTION)
			if dialog1.ShowModal()==wx.ID_YES:
				self.gray_scale=True
				self.text_inputimages.SetLabel('Selected '+str(len(self.path_to_images))+' image(s) in: '+path+' (into gray scale).')
			else:
				self.gray_scale=False
				self.text_inputimages.SetLabel('Selected '+str(len(self.path_to_images))+' image(s) in: '+path+'.')
			dialog1.Destroy()

		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.result_path=dialog.GetPath()
			self.text_outputfolder.SetLabel('Processed images will be in: '+self.result_path+'.')
		dialog.Destroy()


	def downsize_images(self,event):

		dialog=wx.MessageDialog(self,'Proportional downsize the images?','Downsize the images?',wx.YES_NO|wx.ICON_QUESTION)

		if dialog.ShowModal()==wx.ID_YES:
			dialog1=wx.NumberEntryDialog(self,'Enter the desired downsizing factor %','A number between 1 and 99 (%):','Desired downsizing factor %',50,1,99)
			if dialog1.ShowModal()==wx.ID_OK:
				self.downsize_factor=int(dialog1.GetValue())
				self.text_downsizeimages.SetLabel('Proportionally downsize image to '+str(self.downsize_factor)+'%.')
			else:
				self.downsize_factor=None
				self.text_downsizeimages.SetLabel('Not to downsize images.')
			dialog1.Destroy()
		else:
			self.downsize_factor=None
			self.text_downsizeimages.SetLabel('Not to downsize images.')

		dialog.Destroy()


	def crop_images(self,event):

		if self.path_to_images is None:

			wx.MessageBox('No image selected.','Error',wx.OK|wx.ICON_ERROR)

		else:

			extension=os.path.splitext(os.path.basename(self.path_to_images[0]))[1]

			if extension in ['.lif','.LIF']:
				pass
			else:
				image=tifffile.imread(self.path_to_images[0])

			if self.downsize_factor is not None:
				if extension in ['.lif','.LIF']:
					pass
				elif extension in ['.tif','.TIF','.tiff','.TIFF']:
					if len(list(image.shape))<3:
						image=resize(image,(int(image.shape[1]*self.downsize_factor/100),int(image.shape[0]*self.downsize_factor/100)))
					else:
						image=resize(image,(int(image.shape[1]*self.downsize_factor/100),int(image.shape[0]*self.downsize_factor/100),image.shape[2]))
						page=tifffile.TiffFile(self.path_to_images[0]).pages[0]
						num_channels=image.shape[2]
						lut_available=page.photometric==tifffile.PHOTOMETRIC.PALETTE
						rgb_channels=[]
						for c in range(num_channels):
							if lut_available:
								lut=page.colormap
								channel_data=image[...,c]
								rgb_channel=np.zeros((*image.shape,3),dtype=np.uint8)
								for i in range(3):
									rgb_channel[...,i]=lut[i,image]//256
							else:
								cmap=cm.rainbow(c/(num_channels-1))
								rgb_channel=cm.ScalarMappable(cmap=cm.rainbow).to_rgba(image,bytes=True)[:,:,:3]
							rgb_channels.append(rgb_channel)
						image=np.mean(rgb_channels,axis=0).astype(np.uint8)

				elif extension in ['.qptiff','.QPTIFF']:
					if len(list(image.shape))<3:
						image=resize(image,(int(image.shape[1]*self.downsize_factor/100),int(image.shape[0]*self.downsize_factor/100)))
					else:
						image=resize(image,(int(image.shape[2]*self.downsize_factor/100),int(image.shape[1]*self.downsize_factor/100),image.shape[0]))
					page=tifffile.TiffFile(self.path_to_images[0]).pages[0]
					num_channels=image.shape[0]
					lut_available=page.photometric==tifffile.PHOTOMETRIC.PALETTE
					rgb_channels=[]
					for c in range(num_channels):
						if lut_available:
							lut=page.colormap
							channel_data=image[c,...]
							rgb_channel=np.zeros((*image.shape,3),dtype=np.uint8)
							for i in range(3):
								rgb_channel[...,i]=lut[i,image]//256
						else:
							cmap=cm.rainbow(c/(num_channels-1))
							rgb_channel=cm.ScalarMappable(cmap=cm.rainbow).to_rgba(image,bytes=True)[:,:,:3]
						rgb_channels.append(rgb_channel)
					image=np.mean(rgb_channels,axis=0).astype(np.uint8)

				else:
					if len(list(image.shape))<3:
						image=resize(image,(int(image.shape[1]*self.downsize_factor/100),int(image.shape[0]*self.downsize_factor/100)))
					else:
						image=resize(image,(int(image.shape[1]*self.downsize_factor/100),int(image.shape[0]*self.downsize_factor/100),image.shape[2]))

			if self.gray_scale:
				if extension in ['.svs','.SVS']:
					image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
					image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
			
			canvas=np.copy(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
			h,w=image.shape[:2]
			if h>10000:
				interval_h=1000
				scale_h=4
				thickness_h=8
				edge_h=200
			elif h>5000:
				interval_h=500
				scale_h=3
				thickness_h=5
				edge_h=100
			elif h>1000:
				interval_h=200
				scale_h=2
				thickness_h=2
				edge_h=50
			else:
				interval_h=50
				scale_h=0.5
				thickness_h=1
				edge_h=10
			if w>10000:
				interval_w=500
				scale_w=4
				thickness_w=8
				edge_w=200
			elif w>5000:
				interval_w=500
				scale_w=3
				thickness_w=5
				edge_w=100
			elif w>1000:
				interval_w=200
				scale_w=2
				thickness_w=2
				edge_w=50
			else:
				interval_w=50
				scale_w=0.5
				thickness_w=1
				edge_w=10
			for y in range(0,h,interval_h):
				cv2.line(canvas,(0,y),(w,y),(255,0,255),thickness_h)
				cv2.putText(canvas,str(y),(edge_h,y+edge_h),cv2.FONT_HERSHEY_SIMPLEX,scale_h,(255,0,255),thickness_h)
			for x in range(0,w,interval_w):
				cv2.line(canvas,(x,0),(x,h),(255,0,255),thickness_w)
				cv2.putText(canvas,str(x),(x+edge_w,edge_w),cv2.FONT_HERSHEY_SIMPLEX,scale_w,(255,0,255),thickness_w)
			cv2.namedWindow('The first image in coordinates',cv2.WINDOW_NORMAL)
			cv2.imshow('The first image in coordinates',canvas)

			stop=False
			while stop is False:
				dialog=wx.TextEntryDialog(self,'Enter the coordinates (integers) of the cropping window','Format:[left,right,top,bottom]')
				if dialog.ShowModal()==wx.ID_OK:
					coordinates=list(dialog.GetValue().split(','))
					if len(coordinates)==4:
						try:
							self.left=int(coordinates[0])
							self.right=int(coordinates[1])
							self.top=int(coordinates[2])
							self.bottom=int(coordinates[3])
							self.crop_image=True
							stop=True
							self.text_cropimage.SetLabel('The cropping window is from left: '+str(self.left)+' to right: '+str(self.right)+', from top: '+str(self.top)+' to bottom: '+str(self.bottom)+'.')
						except:
							self.crop_image=False
							wx.MessageBox('Please enter 4 integers.','Error',wx.OK|wx.ICON_ERROR)
							self.text_cropimage.SetLabel('Not to crop the images')
					else:
						self.crop_image=False
						wx.MessageBox('Please enter the coordinates (integers) in correct format.','Error',wx.OK|wx.ICON_ERROR)
						self.text_cropimage.SetLabel('Not to crop the images')
				else:
					self.crop_image=False
					self.text_cropimage.SetLabel('Not to crop the images')
					stop=True
				dialog.Destroy()

			cv2.destroyAllWindows()
			

	def enhance_contrasts(self,event):

		if self.path_to_images is None:

			wx.MessageBox('No image selected.','Error',wx.OK|wx.ICON_ERROR)

		else:

			extension=os.path.splitext(os.path.basename(self.path_to_images[0]))[1]
			image=tifffile.imread(self.path_to_images[0])

			if self.downsize_factor is not None:
				if len(list(image.shape))<3:
					image=resize(image,(int(image.shape[1]*self.downsize_factor/100),int(image.shape[0]*self.downsize_factor/100)))
				else:
					image=resize(image,(int(image.shape[1]*self.downsize_factor/100),int(image.shape[0]*self.downsize_factor/100),image.shape[2]))

			if self.gray_scale:
				if extension in ['.svs','.SVS']:
					image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
					image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

			image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

			stop=False
			while stop is False:
				cv2.destroyAllWindows()
				cv2.namedWindow('The first image in coordinates',cv2.WINDOW_NORMAL)
				cv2.imshow('The first image in coordinates',image)
				dialog=wx.TextEntryDialog(self,'Enter the fold changes for contrast enhancement','A number between 1.0~5.0')
				if dialog.ShowModal()==wx.ID_OK:
					contrast=dialog.GetValue()
					try:
						self.contrast=float(contrast)
						show_image=image*self.contrast
						show_image[show_image>255]=255
						show_image=np.uint8(show_image)
						cv2.destroyAllWindows()
						cv2.namedWindow('The first image in coordinates',cv2.WINDOW_NORMAL)
						cv2.imshow('The first image in coordinates',show_image)
						dialog1=wx.MessageDialog(self,'Apply the current contrast value?','Apply value?',wx.YES_NO|wx.ICON_QUESTION)
						if dialog1.ShowModal()==wx.ID_YES:
							stop=True
							self.enhance_contrast=True
							self.text_enhancecontrast.SetLabel('The contrast enhancement fold change is: '+str(self.contrast)+'.')
						else:
							self.enhance_contrast=False
							self.text_enhancecontrast.SetLabel('Not to enhance contrast.')
						dialog1.Destroy()
					except:
						self.enhance_contrast=False
						wx.MessageBox('Please enter a float number between 1.0~5.0.','Error',wx.OK|wx.ICON_ERROR)
						self.text_enhancecontrast.SetLabel('Not to enhance contrast.')
				else:
					self.enhance_contrast=False
					stop=True
					self.text_enhancecontrast.SetLabel('Not to enhance contrast.')
				dialog.Destroy()
			cv2.destroyAllWindows()


	def preprocess_images(self,event):

		if self.path_to_images is None or self.result_path is None:

			wx.MessageBox('No input image(s) / output folder.','Error',wx.OK|wx.ICON_ERROR)

		else:

			print('Start to preprocess image(s)...')

			for i in self.path_to_images:
				preprocess_image(i,self.result_path,self.downsize_factor,enhance_contrast=self.enhance_contrast,contrast=self.contrast,
					crop_image=self.crop_image,left=self.left,right=self.right,top=self.top,bottom=self.bottom,gray_scale=self.gray_scale)

			print('Preprocessing completed!')



class WindowLv2_DrawMarkers(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_DrawMarkers,self).__init__(parent=None,title=title,size=(1000,220))
		self.path_to_videos=None # path to a batch of videos for marker drawing
		self.framewidth=None # if not None, will resize the video frame keeping the original w:h ratio
		self.result_path=None # the folder for storing videos with markers

		self.display_window()


	def display_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_inputvideos=wx.BoxSizer(wx.HORIZONTAL)
		button_inputvideos=wx.Button(panel,label='Select the video(s)\nfor marker drawing',size=(300,40))
		button_inputvideos.Bind(wx.EVT_BUTTON,self.select_videos)
		wx.Button.SetToolTip(button_inputvideos,'Select one or more videos. Common video formats (mp4, mov, avi, m4v, mkv, mpg, mpeg) are supported except wmv format.')
		self.text_inputvideos=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_inputvideos.Add(button_inputvideos,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_inputvideos.Add(self.text_inputvideos,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_inputvideos,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe videos with markers',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'Videos with markers will be stored in the selected folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_preprocessvideos=wx.Button(panel,label='Start to draw markers',size=(300,40))
		button_preprocessvideos.Bind(wx.EVT_BUTTON,self.draw_markers)
		wx.Button.SetToolTip(button_preprocessvideos,'Draw markers in videos.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_preprocessvideos,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_videos(self,event):

		wildcard='Video files(*.avi;*.mpg;*.mpeg;*.wmv;*.mp4;*.mkv;*.m4v;*.mov;*.mts)|*.avi;*.mpg;*.mpeg;*.wmv;*.mp4;*.mkv;*.m4v;*.mov;*.mts'
		dialog=wx.FileDialog(self,'Select video(s)','','',wildcard,style=wx.FD_MULTIPLE)

		if dialog.ShowModal()==wx.ID_OK:

			self.path_to_videos=dialog.GetPaths()
			self.path_to_videos.sort()
			path=os.path.dirname(self.path_to_videos[0])
			dialog1=wx.MessageDialog(self,'Proportional resize the video frames?','(Optional) resize the frames?',wx.YES_NO|wx.ICON_QUESTION)
			if dialog1.ShowModal()==wx.ID_YES:
				dialog2=wx.NumberEntryDialog(self,'Enter the desired frame width','The unit is pixel:','Desired frame width',480,1,10000)
				if dialog2.ShowModal()==wx.ID_OK:
					self.framewidth=int(dialog2.GetValue())
					if self.framewidth<10:
						self.framewidth=10
					self.text_inputvideos.SetLabel('Selected '+str(len(self.path_to_videos))+' video(s) in: '+path+' (proportionally resize framewidth to '+str(self.framewidth)+').')
				else:
					self.framewidth=None
					self.text_inputvideos.SetLabel('Selected '+str(len(self.path_to_videos))+' video(s) in: '+path+' (original framesize).')
				dialog2.Destroy()
			else:
				self.framewidth=None
				self.text_inputvideos.SetLabel('Selected '+str(len(self.path_to_videos))+' video(s) in: '+path+' (original framesize).')
			dialog1.Destroy()

		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.result_path=dialog.GetPath()
			self.text_outputfolder.SetLabel('Videos with markers will be in: '+self.result_path+'.')
		dialog.Destroy()


	def draw_markers(self,event):

		if self.path_to_videos is None or self.result_path is None:
			wx.MessageBox('No input video(s) / output folder.','Error',wx.OK|wx.ICON_ERROR)
		else:
			WindowLv3_DrawMarkers(self.path_to_videos,self.result_path,framewidth=self.framewidth)



class WindowLv3_DrawMarkers(wx.Frame):

	def __init__(self,path_to_videos,result_path,framewidth=None):

		super().__init__(parent=None,title='Draw Makers in Videos')

		self.path_to_videos=path_to_videos
		self.result_path=result_path
		self.framewidth=framewidth
		capture=cv2.VideoCapture(self.path_to_videos[0])
		while True:
			retval,frame=capture.read()
			break
		capture.release()
		if self.framewidth is not None:
			self.image=cv2.resize(frame,(self.framewidth,int(frame.shape[0]*self.framewidth/frame.shape[1])),interpolation=cv2.INTER_AREA)
		else:
			self.image=frame

		self.draw_lines=False

		self.lines=[]
		self.current_line=None
		self.circles=[]
		self.current_circle=None

		self.current_color=(255,0,0)
		self.thickness=max(1,round((self.image.shape[0]+self.image.shape[1])/320))

		self.panel=wx.Panel(self)

		main_sizer=wx.BoxSizer(wx.VERTICAL)

		self.image_panel=wx.Panel(self.panel)
		self.image_panel.Bind(wx.EVT_PAINT,self.on_paint)
		self.image_panel.Bind(wx.EVT_LEFT_DOWN,self.on_left_down)
		self.image_panel.Bind(wx.EVT_LEFT_UP,self.on_left_up)
		self.image_panel.Bind(wx.EVT_MOTION,self.on_motion)

		button_sizer=wx.BoxSizer(wx.HORIZONTAL)

		shape_button=wx.Button(self.panel,label='Select Shape')
		shape_button.Bind(wx.EVT_BUTTON,self.on_select_shape)

		color_button=wx.Button(self.panel,label='Select Color')
		color_button.Bind(wx.EVT_BUTTON,self.on_select_color)

		undo_button=wx.Button(self.panel,label='Undo Drawing')
		undo_button.Bind(wx.EVT_BUTTON,self.on_undo)

		draw_button=wx.Button(self.panel,label='Draw Markers')
		draw_button.Bind(wx.EVT_BUTTON,self.draw_markers)

		button_sizer.Add(shape_button,0,wx.ALIGN_CENTER|wx.ALL,5)
		button_sizer.Add(color_button,0,wx.ALIGN_CENTER|wx.ALL,5)
		button_sizer.Add(undo_button,0,wx.ALIGN_CENTER|wx.ALL,5)
		button_sizer.Add(draw_button,0,wx.ALIGN_CENTER|wx.ALL,5)

		main_sizer.Add(self.image_panel,1,wx.EXPAND)
		main_sizer.Add(button_sizer,0,wx.ALIGN_CENTER|wx.ALL,5)

		self.panel.SetSizer(main_sizer)

		self.SetSize((self.image.shape[1],self.image.shape[0]+50))
		self.Centre()
		self.Show()


	def on_paint(self,event):

		dc=wx.PaintDC(self.image_panel)
		dc.Clear()

		image=self.image.copy()

		for line in self.lines:
			self.draw_line(image,line)
		if self.current_line:
			self.draw_line(image,self.current_line)

		for circle in self.circles:
			self.draw_circle(image,circle)
		if self.current_circle:
			self.draw_circle(image,self.current_circle)

		height,width=image.shape[:2]
		image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		bitmap=wx.Bitmap.FromBuffer(width,height,image_rgb)
		dc.DrawBitmap(bitmap,0,0,False)


	def on_left_down(self,event):

		x,y=event.GetPosition()

		if self.draw_lines:
			self.current_line={'start':(x,y),'end':(x,y),'color':self.current_color}
		else:
			self.current_circle={'start':(x,y),'end':(x,y),'color':self.current_color}


	def on_left_up(self,event):

		if self.draw_lines:

			if self.current_line:
				x,y=event.GetPosition()
				self.current_line['end']=(x,y)
				self.lines.append(self.current_line)
				self.current_line=None
				self.panel.Refresh()

		else:

			if self.current_circle:
				x,y=event.GetPosition()
				self.current_circle['end']=(x,y)
				self.circles.append(self.current_circle)
				self.current_circle=None
				self.panel.Refresh()


	def on_motion(self,event):

		if self.draw_lines:

			if event.Dragging() and event.LeftIsDown() and self.current_line:
				x,y=event.GetPosition()
				self.current_line['end']=(x,y)
				self.panel.Refresh()

		else:

			if event.Dragging() and event.LeftIsDown() and self.current_circle:
				x,y=event.GetPosition()
				self.current_circle['end']=(x,y)
				self.panel.Refresh()


	def draw_line(self,image,line):

		start=line['start']
		end=line['end']
		color=line['color']

		overlay=image.copy()
		cv2.line(overlay,start,end,color,self.thickness)
		alpha=1.0
		cv2.addWeighted(overlay,alpha,image,1-alpha,0,image)


	def draw_circle(self,image,circle):

		start=circle['start']
		end=circle['end']
		color=circle['color']
		radius=int(((end[0]-start[0])**2+(end[1]-start[1])**2)**0.5)
		center=start

		overlay=image.copy()
		cv2.circle(overlay,center,radius,color,self.thickness)
		alpha=1.0
		cv2.addWeighted(overlay,alpha,image,1-alpha,0,image)


	def on_undo(self,event):

		if self.draw_lines:

			if self.lines:
				self.lines.pop()
				self.panel.Refresh()

		else:

			if self.circles:
				self.circles.pop()
				self.panel.Refresh()


	def on_select_shape(self,event):

		dialog=wx.MessageDialog(self,'Draw lines? If not, will draw circles','Draw lines?',wx.YES_NO|wx.ICON_QUESTION)
		if dialog.ShowModal()==wx.ID_YES:
			self.draw_lines=True
		else:
			self.draw_lines=False
		dialog.Destroy()


	def on_select_color(self,event):

		color_data=wx.ColourData()
		color_dialog=wx.ColourDialog(self,color_data)

		if color_dialog.ShowModal()==wx.ID_OK:
			color_data=color_dialog.GetColourData()
			color=color_data.GetColour()
			self.current_color=(color.Blue(),color.Green(),color.Red())


	def draw_markers(self,event):

		if len(self.circles)==0 and len(self.lines)==0:

			wx.MessageBox('No Markers.','Error',wx.OK|wx.ICON_ERROR)

		else:

			for i in self.path_to_videos:

				capture=cv2.VideoCapture(i)
				name=os.path.basename(i).split('.')[0]
				fps=round(capture.get(cv2.CAP_PROP_FPS))
				num_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
				width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
				height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

				if self.framewidth is not None:
					w=int(self.framewidth)
					h=int(self.framewidth*height/width)
				else:
					w=width
					h=height

				thickness=max(1,round((w+h)/320))

				writer=cv2.VideoWriter(os.path.join(self.result_path,name+'_marked.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h),True)

				while True:

					ret,frame=capture.read()

					if frame is None:
						break

					if self.framewidth is not None:
						frame=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)

					for line in self.lines:
						start=line['start']
						end=line['end']
						color=line['color']
						cv2.line(frame,start,end,color,thickness)

					for circle in self.circles:
						start=circle['start']
						end=circle['end']
						color=circle['color']
						radius=int(((end[0]-start[0])**2+(end[1]-start[1])**2)**0.5)
						center=start
						cv2.circle(frame,center,radius,color,thickness)

					writer.write(np.uint8(frame))

				writer.release()
				capture.release()


			print('Marker Drawing completed!')



class WindowLv1_TrainingModule(wx.Frame):

	def __init__(self,title):

		super(WindowLv1_TrainingModule,self).__init__(parent=None,title=title,size=(500,350))
		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,60,0)

		button_generateimages=wx.Button(panel,label='Generate Image Examples',size=(300,40))
		button_generateimages.Bind(wx.EVT_BUTTON,self.generate_images)
		wx.Button.SetToolTip(button_generateimages,
			'Extract images from LIF/TIF/SVS/QPTIFF files for annotation of the cells of your interest. See Extended Guide for how to select images to annotate.')
		boxsizer.Add(button_generateimages,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		link_annotate=wx.lib.agw.hyperlink.HyperLinkCtrl(panel,0,'\nAnnotate images with EZannot\n',URL='https://github.com/yujiahu415/EZannot')
		boxsizer.Add(link_annotate,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		button_traindetectors=wx.Button(panel,label='Train Detectors',size=(300,40))
		button_traindetectors.Bind(wx.EVT_BUTTON,self.train_detectors)
		wx.Button.SetToolTip(button_traindetectors,
			'The trained Detectors can detect the cells of your interest. See Extended Guide for how to set parameters for training.')
		boxsizer.Add(button_traindetectors,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		button_testdetectors=wx.Button(panel,label='Test Detectors',size=(300,40))
		button_testdetectors.Bind(wx.EVT_BUTTON,self.test_detectors)
		wx.Button.SetToolTip(button_testdetectors,
			'Test trained Detectors on the annotated ground-truth image dataset (similar to the image dataset used for training a Detector).')
		boxsizer.Add(button_testdetectors,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def generate_images(self,event):

		WindowLv2_GenerateImages('Generate Image Examples')


	def train_detectors(self,event):

		WindowLv2_TrainDetectors('Train Detectors')


	def test_detectors(self,event):

		WindowLv2_TestDetectors('Test Detectors')



class WindowLv1_AnalysisModule(wx.Frame):

	def __init__(self,title):

		super(WindowLv1_AnalysisModule,self).__init__(parent=None,title=title,size=(500,295))
		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,60,0)

		button_analyzemultichannels=wx.Button(panel,label='Analyze Multichannel Images',size=(300,40))
		button_analyzemultichannels.Bind(wx.EVT_BUTTON,self.analyze_multichannels)
		wx.Button.SetToolTip(button_analyzemultichannels,
			'Automatically detect cells of your interest and analyze their numbers, areas, and pixel intensities in multi-channel images.')
		boxsizer.Add(button_analyzemultichannels,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		button_analyzesinglechannel=wx.Button(panel,label='Analyze Singlechannel Images',size=(300,40))
		button_analyzesinglechannel.Bind(wx.EVT_BUTTON,self.analyze_singlechannels)
		wx.Button.SetToolTip(button_analyzesinglechannel,
			'Automatically detect cells of your interest and analyze their numbers, areas, and pixel intensities in single-channel images.')
		boxsizer.Add(button_analyzesinglechannel,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		button_calculateintensities=wx.Button(panel,label='Calculate Channel Intensities',size=(300,40))
		button_calculateintensities.Bind(wx.EVT_BUTTON,self.calculate_intensities)
		wx.Button.SetToolTip(button_calculateintensities,
			'Calculate total intensity of each channel in images.')
		boxsizer.Add(button_calculateintensities,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def analyze_multichannels(self,event):

		WindowLv2_AnalyzeMultiChannels('Analyze Multichannel Images')


	def analyze_singlechannels(self,event):

		WindowLv2_AnalyzeSingleChannel('Analyze Singlechannel Images')


	def calculate_intensities(self,event):

		WindowLv2_CalculateTotalIntensity('Calculate Channel Intensities')







class WindowLv2_GenerateImages(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_GenerateImages,self).__init__(parent=None,title=title,size=(1000,240))
		self.path_to_files=None
		self.result_path=None
		self.fov_dim=1280
		self.black_background=True

		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_inputfiles=wx.BoxSizer(wx.HORIZONTAL)
		button_inputfiles=wx.Button(panel,label='Select the LIF/TIF/SVS/QPTIFF file(s)\nto generate image examples',size=(300,40))
		button_inputfiles.Bind(wx.EVT_BUTTON,self.select_files)
		wx.Button.SetToolTip(button_inputfiles,'Select one or more *.LIF or *.TIF or *.SVS or *.QPTIFF files.')
		self.text_inputfiles=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_inputfiles.Add(button_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_inputfiles.Add(self.text_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store the\ngenerated image examples',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'The generated image examples (extracted frames) will be stored in this folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_fov=wx.BoxSizer(wx.HORIZONTAL)
		button_fov=wx.Button(panel,label='Specify the dimension of one\nfield of view in images to analyze',size=(300,40))
		button_fov.Bind(wx.EVT_BUTTON,self.specify_fov)
		wx.Button.SetToolTip(button_fov,'Specify the width of one field of view (square shape), the images to analyze will be divided into smaller fields of view according to the specified dimension.')
		self.text_fov=wx.StaticText(panel,label='Default: 1280',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_fov.Add(button_fov,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_fov.Add(self.text_fov,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_fov,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		generate=wx.BoxSizer(wx.HORIZONTAL)
		button_generate=wx.Button(panel,label='Start to generate image examples',size=(300,40))
		button_generate.Bind(wx.EVT_BUTTON,self.generate_images)
		wx.Button.SetToolTip(button_generate,'Press the button to start generating image examples.')
		generate.Add(button_generate,0,wx.LEFT,50)
		boxsizer.Add(0,5,0)
		boxsizer.Add(generate,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_files(self,event):

		wildcard='LIF/TIF/SVS/QPTIFF files (*.lif/*.tif/*.svs/*.qptiff)|*.lif;*.LIF;*.tif;*.TIF;*.tiff;*.TIFF;*.svs;*.SVS;*.qptiff;*.QPTIFF'
		dialog=wx.FileDialog(self,'Select LIF/TIF/SVS/QPTIFF file(s)','','',wildcard,style=wx.FD_MULTIPLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_files=dialog.GetPaths()
			path=os.path.dirname(self.path_to_files[0])
			self.text_inputfiles.SetLabel('Selected '+str(len(self.path_to_files))+' file(s) in: '+path+'.')
		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.result_path=dialog.GetPath()
			self.text_outputfolder.SetLabel('Generate image examples in: '+self.result_path+'.')
		dialog.Destroy()


	def specify_fov(self,event):

		dialog=wx.NumberEntryDialog(self,'Enter the number of sections\nthe width and height should be divided','Enter a number:','Number of field of view',1280,1,2048)
		if dialog.ShowModal()==wx.ID_OK:
			self.fov_dim=int(dialog.GetValue())
		else:
			self.fov_dim=1280
		dialog.Destroy()
		if self.fov_dim<128:
			self.fov_dim=128

		dialog=wx.MessageDialog(self,'Is the background in the images black/darker?','Darker background?',wx.YES_NO|wx.ICON_QUESTION)
		if dialog.ShowModal()==wx.ID_YES:
			self.black_background=True
			self.text_fov.SetLabel('The dimension of one field of view : '+str(self.fov_dim)+' X '+str(self.fov_dim)+' (background darker).')
		else:
			self.black_background=False
			self.text_fov.SetLabel('The dimension of one field of view : '+str(self.fov_dim)+' X '+str(self.fov_dim)+' (background lighter).')
		dialog.Destroy()


	def generate_images(self,event):

		if self.path_to_files is None or self.result_path is None:

			wx.MessageBox('No input file(s) / output folder selected.','Error',wx.OK|wx.ICON_ERROR)

		else:

			print('Generating image examples...')
			for i in self.path_to_files:
				extract_images(i,self.result_path,self.fov_dim,black_background=self.black_background)
			print('Image example generation completed!')



class WindowLv2_TrainDetectors(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_TrainDetectors,self).__init__(parent=None,title=title,size=(1000,280))
		self.path_to_trainingimages=None
		self.path_to_annotation=None
		self.num_rois=128
		self.inference_size=None
		self.black_background=None
		self.iteration_num=5000
		self.detector_path=os.path.join(the_absolute_current_path,'detectors')
		self.path_to_detector=None

		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_selectimages=wx.BoxSizer(wx.HORIZONTAL)
		button_selectimages=wx.Button(panel,label='Select the folder containing\nall the training images',size=(300,40))
		button_selectimages.Bind(wx.EVT_BUTTON,self.select_images)
		wx.Button.SetToolTip(button_selectimages,'The folder that stores all the training images.')
		self.text_selectimages=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectimages.Add(button_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectimages.Add(self.text_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectannotation=wx.BoxSizer(wx.HORIZONTAL)
		button_selectannotation=wx.Button(panel,label='Select the *.json\nannotation file',size=(300,40))
		button_selectannotation.Bind(wx.EVT_BUTTON,self.select_annotation)
		wx.Button.SetToolTip(button_selectannotation,'The .json file that stores the annotation for the training images. Should be in “COCO instance segmentation” format.')
		self.text_selectannotation=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectannotation.Add(button_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectannotation.Add(self.text_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_background=wx.BoxSizer(wx.HORIZONTAL)
		button_background=wx.Button(panel,label='Specify whether the background is\nblack/darker in training images',size=(300,40))
		button_background.Bind(wx.EVT_BUTTON,self.specify_background)
		wx.Button.SetToolTip(button_background,'This helps the trained Detector to make up the missing regions when analyzing images with the fixed field of view.')
		self.text_background=wx.StaticText(panel,label='Not specified.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_background.Add(button_background,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_background.Add(self.text_background,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_background,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_iterations=wx.BoxSizer(wx.HORIZONTAL)
		button_iterations=wx.Button(panel,label='Specify the iteration number\nfor the Detector training',size=(300,40))
		button_iterations.Bind(wx.EVT_BUTTON,self.input_iterations)
		wx.Button.SetToolTip(button_iterations,'More training iterations typically yield higher accuracy but take longer.')
		self.text_iterations=wx.StaticText(panel,label='Default: 5000.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_iterations.Add(button_iterations,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_iterations.Add(self.text_iterations,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_iterations,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_train=wx.Button(panel,label='Train the Detector',size=(300,40))
		button_train.Bind(wx.EVT_BUTTON,self.train_detector)
		wx.Button.SetToolTip(button_train,'English letters, numbers, “_”, or “-” are acceptable for the names but no “@” or “^”.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_train,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_images(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_trainingimages=dialog.GetPath()
			self.text_selectimages.SetLabel('Path to training images: '+self.path_to_trainingimages+'.')
		dialog.Destroy()


	def select_annotation(self,event):

		wildcard='Annotation File (*.json)|*.json'
		dialog=wx.FileDialog(self, 'Select the annotation file (.json)','',wildcard=wildcard,style=wx.FD_OPEN)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_annotation=dialog.GetPath()
			f=open(self.path_to_annotation)
			info=json.load(f)
			classnames=[]
			for i in info['categories']:
				if i['id']>0:
					classnames.append(i['name'])
			self.text_selectannotation.SetLabel('Cell categories in annotation file: '+str(classnames)+'.')
		dialog.Destroy()


	def specify_background(self,event):

		dialog=wx.MessageDialog(self,'Is the background in the images black/darker?','Darker background?',wx.YES_NO|wx.ICON_QUESTION)
		if dialog.ShowModal()==wx.ID_YES:
			self.black_background=0
			self.text_background.SetLabel('The background in images is black/darker.')
		else:
			self.black_background=1
			self.text_background.SetLabel('The background in images is white/lighter.')
		dialog.Destroy()


	def input_iterations(self,event):

		dialog=wx.NumberEntryDialog(self,'Input the iteration number\nfor the Detector training','Enter a number:','Iterations',5000,1,1000000)
		if dialog.ShowModal()==wx.ID_OK:
			self.iteration_num=int(dialog.GetValue())
			self.text_iterations.SetLabel('Training iteration number: '+str(self.iteration_num)+'.')
		dialog.Destroy()


	def train_detector(self,event):

		if self.path_to_trainingimages is None or self.path_to_annotation is None or self.black_background is None:

			wx.MessageBox('No training images / annotation file / background in images specified.','Error',wx.OK|wx.ICON_ERROR)

		else:

			cell_sizes=['Sparse and large (e.g., large tissue areas)','Median (e.g., structures formed by group of cells)','Small (e.g. typical cell bodies)','Extremely small (e.g., dense subcellular structures)']
			dialog=wx.SingleChoiceDialog(self,message='How large are the objects to detect\ncompared to the images?',caption='Object size',choices=cell_sizes)
			if dialog.ShowModal()==wx.ID_OK:
				cell_size=dialog.GetStringSelection()
				if cell_size=='Sparse and large (e.g., large tissue areas)':
					self.num_rois=128
				elif cell_size=='Median (e.g., structures formed by group of cells)':
					self.num_rois=256
				elif cell_size=='Small (e.g. typical cell bodies)':
					self.num_rois=512
				else:
					self.num_rois=1024
			dialog.Destroy()

			images=[i for i in os.listdir(self.path_to_trainingimages) if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.tif') or i.endswith('.tiff')]
			self.inference_size=int(cv2.imread(os.path.join(self.path_to_trainingimages,images[0])).shape[1])

			do_nothing=False
			stop=False
			while stop is False:
				dialog=wx.TextEntryDialog(self,'Enter a name for the Detector to train','Detector name')
				if dialog.ShowModal()==wx.ID_OK:
					if dialog.GetValue()!='':
						self.path_to_detector=os.path.join(self.detector_path,dialog.GetValue())
						if not os.path.isdir(self.path_to_detector):
							stop=True
						else:
							wx.MessageBox('The name already exists.','Error',wx.OK|wx.ICON_ERROR)
				else:
					do_nothing=True
					stop=True
				dialog.Destroy()

			if do_nothing is False:
				DT=Detector()
				DT.train(self.path_to_annotation,self.path_to_trainingimages,self.path_to_detector,self.iteration_num,self.inference_size,self.num_rois,black_background=self.black_background)



class WindowLv2_TestDetectors(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_TestDetectors,self).__init__(parent=None,title=title,size=(1000,280))
		self.path_to_testingimages=None
		self.path_to_annotation=None
		self.detector_path=os.path.join(the_absolute_current_path,'detectors')
		self.path_to_detector=None
		self.output_path=None

		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_selectdetector=wx.BoxSizer(wx.HORIZONTAL)
		button_selectdetector=wx.Button(panel,label='Select a Detector\nto test',size=(300,40))
		button_selectdetector.Bind(wx.EVT_BUTTON,self.select_detector)
		wx.Button.SetToolTip(button_selectdetector,'The names of cells in the testing dataset should match those in the selected Detector.')
		self.text_selectdetector=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectdetector.Add(button_selectdetector,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectdetector.Add(self.text_selectdetector,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_selectdetector,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectimages=wx.BoxSizer(wx.HORIZONTAL)
		button_selectimages=wx.Button(panel,label='Select the folder containing\nall the testing images',size=(300,40))
		button_selectimages.Bind(wx.EVT_BUTTON,self.select_images)
		wx.Button.SetToolTip(button_selectimages,'The folder that stores all the testing images.')
		self.text_selectimages=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectimages.Add(button_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectimages.Add(self.text_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectannotation=wx.BoxSizer(wx.HORIZONTAL)
		button_selectannotation=wx.Button(panel,label='Select the *.json\nannotation file',size=(300,40))
		button_selectannotation.Bind(wx.EVT_BUTTON,self.select_annotation)
		wx.Button.SetToolTip(button_selectannotation,'The .json file that stores the annotation for the testing images. Should be in “COCO instance segmentation” format.')
		self.text_selectannotation=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectannotation.Add(button_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectannotation.Add(self.text_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectoutpath=wx.BoxSizer(wx.HORIZONTAL)
		button_selectoutpath=wx.Button(panel,label='Select the folder to\nstore testing results',size=(300,40))
		button_selectoutpath.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_selectoutpath,'The folder will stores the testing results.')
		self.text_selectoutpath=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectoutpath.Add(button_selectoutpath,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectoutpath.Add(self.text_selectoutpath,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectoutpath,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		testanddelete=wx.BoxSizer(wx.HORIZONTAL)
		button_test=wx.Button(panel,label='Test the Detector',size=(300,40))
		button_test.Bind(wx.EVT_BUTTON,self.test_detector)
		wx.Button.SetToolTip(button_test,'Test the selected Detector on the annotated, ground-truth testing images.')
		button_delete=wx.Button(panel,label='Delete a Detector',size=(300,40))
		button_delete.Bind(wx.EVT_BUTTON,self.remove_detector)
		wx.Button.SetToolTip(button_delete,'Permanently delete a Detector. The deletion CANNOT be restored.')
		testanddelete.Add(button_test,0,wx.RIGHT,50)
		testanddelete.Add(button_delete,0,wx.LEFT,50)
		boxsizer.Add(0,5,0)
		boxsizer.Add(testanddelete,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_detector(self,event):

		detectors=[i for i in os.listdir(self.detector_path) if os.path.isdir(os.path.join(self.detector_path,i))]
		if '__pycache__' in detectors:
			detectors.remove('__pycache__')
		if '__init__' in detectors:
			detectors.remove('__init__')
		if '__init__.py' in detectors:
			detectors.remove('__init__.py')
		detectors.sort()

		dialog=wx.SingleChoiceDialog(self,message='Select a Detector to test',caption='Test a Detector',choices=detectors)
		if dialog.ShowModal()==wx.ID_OK:
			detector=dialog.GetStringSelection()
			self.path_to_detector=os.path.join(self.detector_path,detector)
			cellmapping=os.path.join(self.path_to_detector,'model_parameters.txt')
			with open(cellmapping) as f:
				model_parameters=f.read()
			cell_names=json.loads(model_parameters)['cell_names']
			self.text_selectdetector.SetLabel('Selected: '+str(detector)+' (cells: '+str(cell_names)+').')
		dialog.Destroy()


	def select_images(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_testingimages=dialog.GetPath()
			self.text_selectimages.SetLabel('Path to testing images: '+self.path_to_testingimages+'.')
		dialog.Destroy()


	def select_annotation(self,event):

		wildcard='Annotation File (*.json)|*.json'
		dialog=wx.FileDialog(self, 'Select the annotation file (.json)','',wildcard=wildcard,style=wx.FD_OPEN)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_annotation=dialog.GetPath()
			f=open(self.path_to_annotation)
			info=json.load(f)
			classnames=[]
			for i in info['categories']:
				if i['id']>0:
					classnames.append(i['name'])
			self.text_selectannotation.SetLabel('Cell categories in annotation file: '+str(classnames)+'.')
		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.output_path=dialog.GetPath()
			self.text_selectoutpath.SetLabel('Path to testing images: '+self.output_path+'.')
		dialog.Destroy()


	def test_detector(self,event):

		if self.path_to_detector is None or self.path_to_testingimages is None or self.path_to_annotation is None or self.output_path is None:
			wx.MessageBox('No Detector / training images / annotation file / output path selected.','Error',wx.OK|wx.ICON_ERROR)
		else:
			DT=Detector()
			DT.test(self.path_to_annotation,self.path_to_testingimages,self.path_to_detector,self.output_path)


	def remove_detector(self,event):

		detectors=[i for i in os.listdir(self.detector_path) if os.path.isdir(os.path.join(self.detector_path,i))]
		if '__pycache__' in detectors:
			detectors.remove('__pycache__')
		if '__init__' in detectors:
			detectors.remove('__init__')
		if '__init__.py' in detectors:
			detectors.remove('__init__.py')
		detectors.sort()

		dialog=wx.SingleChoiceDialog(self,message='Select a Detector to delete',caption='Delete a Detector',choices=detectors)
		if dialog.ShowModal()==wx.ID_OK:
			detector=dialog.GetStringSelection()
			dialog1=wx.MessageDialog(self,'Delete '+str(detector)+'?','CANNOT be restored!',wx.YES_NO|wx.ICON_QUESTION)
			if dialog1.ShowModal()==wx.ID_YES:
				shutil.rmtree(os.path.join(self.detector_path,detector))
			dialog1.Destroy()
		dialog.Destroy()



class WindowLv2_AnalyzeMultiChannels(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_AnalyzeMultiChannels,self).__init__(parent=None,title=title,size=(1000,390))
		self.detector_path=None
		self.path_to_detector=None
		self.cell_kinds=None
		self.path_to_files=None
		self.result_path=None
		self.detection_threshold=None
		self.expansion=None
		self.fov_dim=1280
		self.names_colors=None
		self.detection_channel=0
		self.analysis_channels=[]
		self.black_background=True
		self.show_ids=False
		self.filters={}
		self.inners=None
		
		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_inputfiles=wx.BoxSizer(wx.HORIZONTAL)
		button_inputfiles=wx.Button(panel,label='Select the LIF/TIF/QPTIFF file(s)\nfor analyzing cells',size=(300,40))
		button_inputfiles.Bind(wx.EVT_BUTTON,self.select_files)
		wx.Button.SetToolTip(button_inputfiles,'Select one or more *.LIF or *.TIF or *.QPTIFF file(s).')
		self.text_inputfiles=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_inputfiles.Add(button_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_inputfiles.Add(self.text_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe analysis results',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'Will create a subfolder for each file in the selected folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_detection=wx.BoxSizer(wx.HORIZONTAL)
		button_detection=wx.Button(panel,label='Select the Detector to\ndetect cells',size=(300,40))
		button_detection.Bind(wx.EVT_BUTTON,self.select_detector)
		wx.Button.SetToolTip(button_detection,'A trained Detector can detect cells of your interest.')
		self.text_detection=wx.StaticText(panel,label='None',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_detection.Add(button_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_detection.Add(self.text_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_expansion=wx.BoxSizer(wx.HORIZONTAL)
		button_expansion=wx.Button(panel,label='Specify the expansion\nof a cell',size=(300,40))
		button_expansion.Bind(wx.EVT_BUTTON,self.specify_expansion)
		wx.Button.SetToolTip(button_expansion,'If the Detector detects the cell outlines, enter 1. If the Detector detects the nuclei, enter a number > 1 to detect areas outside the nuclei.')
		self.text_expansion=wx.StaticText(panel,label='Default: 1',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_expansion.Add(button_expansion,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_expansion.Add(self.text_expansion,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_expansion,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_channels=wx.BoxSizer(wx.HORIZONTAL)
		button_channels=wx.Button(panel,label='Specify the channels for\ndetection and analysis',size=(300,40))
		button_channels.Bind(wx.EVT_BUTTON,self.specify_channels)
		wx.Button.SetToolTip(button_channels,'Specify the channel used for detecting the cells and those used for analyzing the pixel intensity of the cells')
		self.text_channels=wx.StaticText(panel,label='Default: detection channel: 0; analysis channels: all channels',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_channels.Add(button_channels,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_channels.Add(self.text_channels,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_channels,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_analyze=wx.Button(panel,label='Start to analyze cells',size=(300,40))
		button_analyze.Bind(wx.EVT_BUTTON,self.analyze_cells)
		wx.Button.SetToolTip(button_analyze,'Will output the numbers, areas, and pixel intensities for each cell of interest.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_analyze,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_files(self,event):

		wildcard='LIF/TIF/QPTIFF files (*.lif/*.tif/*.qptiff)|*.lif;*.LIF;*.tif;*.TIF;*.tiff;*.TIFF;*.qptiff;*.QPTIFF'
		dialog=wx.FileDialog(self,'Select LIF/TIF/QPTIFF file(s)','','',wildcard,style=wx.FD_MULTIPLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_files=dialog.GetPaths()
			path=os.path.dirname(self.path_to_files[0])
			self.text_inputfiles.SetLabel('Selected '+str(len(self.path_to_files))+' file(s) in: '+path+'.')
		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.result_path=dialog.GetPath()
			self.text_outputfolder.SetLabel('Results will be in: '+self.result_path+'.')
		dialog.Destroy()


	def select_detector(self,event):

		self.detector_path=os.path.join(the_absolute_current_path,'detectors')

		detectors=[i for i in os.listdir(self.detector_path) if os.path.isdir(os.path.join(self.detector_path,i))]
		if '__pycache__' in detectors:
			detectors.remove('__pycache__')
		if '__init__' in detectors:
			detectors.remove('__init__')
		if '__init__.py' in detectors:
			detectors.remove('__init__.py')
		detectors.sort()
		if 'Choose a new directory of the Detector' not in detectors:
			detectors.append('Choose a new directory of the Detector')

		dialog=wx.SingleChoiceDialog(self,message='Select a Detector',caption='Select a Detector',choices=detectors)
		if dialog.ShowModal()==wx.ID_OK:
			detector=dialog.GetStringSelection()
			if detector=='Choose a new directory of the Detector':
				dialog1=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
				if dialog1.ShowModal()==wx.ID_OK:
					self.path_to_detector=dialog1.GetPaths()
				dialog1.Destroy()
			else:
				self.path_to_detector=os.path.join(self.detector_path,detector)
			with open(os.path.join(self.path_to_detector,'model_parameters.txt')) as f:
				model_parameters=f.read()
			cell_names=json.loads(model_parameters)['cell_names']
			self.fov_dim=int(json.loads(model_parameters)['inferencing_framesize'])
			if int(json.loads(model_parameters)['black_background'])==0:
				self.black_background=True
			else:
				self.black_background=False
			if len(cell_names)>1:
				dialog1=wx.MultiChoiceDialog(self,message='Specify which cells involved in analysis',
					caption='Cell kind',choices=cell_names)
				if dialog1.ShowModal()==wx.ID_OK:
					self.cell_kinds=[cell_names[i] for i in dialog1.GetSelections()]
				else:
					self.cell_kinds=cell_names
				dialog1.Destroy()
			else:
				self.cell_kinds=cell_names
			self.names_colors={}
			self.detection_threshold={}
			colors=[str(hex_code) for hex_code in mpl.colors.cnames.values()]
			for color,cell_name in zip(colors,self.cell_kinds):
				self.names_colors[cell_name]=color
			for cell_name in self.cell_kinds:
				dialog1=ColorPicker(self,f'Color for annotating {cell_name}',[cell_name,self.names_colors[cell_name]])
				if dialog1.ShowModal()==wx.ID_OK:
					(r,b,g,_)=dialog1.color_picker.GetColour()
					new_color='#%02x%02x%02x'%(r,b,g)
					self.names_colors[cell_name]=new_color
				dialog1.Destroy()
				dialog1=wx.MessageDialog(self,'Show the IDs for each detected cells?','Show IDs?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog1.ShowModal()==wx.ID_YES:
					self.show_ids=True
				else:
					self.show_ids=False
				dialog1.Destroy()
				dialog1=wx.NumberEntryDialog(self,'Detection threshold for '+str(cell_name),'Enter an number between 0 and 100','Detection threshold for '+str(cell_name),0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					self.detection_threshold[cell_name]=int(dialog1.GetValue())/100
				else:
					self.detection_threshold[cell_name]=0
				dialog1.Destroy()
			self.text_detection.SetLabel('Detector: '+detector+'; '+'The cell kinds / detection threshold: '+str(self.detection_threshold)+'.')
		dialog.Destroy()

		if self.path_to_detector is not None:
			dialog=wx.MessageDialog(self,'Detect the outlines of inner structures?\n(Can be useful to calculate, e.g. G-ratios)','Detect inner structures?',wx.YES_NO|wx.ICON_QUESTION)
			if dialog.ShowModal()==wx.ID_YES:
				dialog1=wx.MessageDialog(self,'Is the inner structures brighter?','Brighter inner structures?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog1.ShowModal()==wx.ID_YES:
					self.inners='white'
				else:
					self.inners='black'
				dialog1.Destroy()
			else:
				self.inners=None
			dialog.Destroy()


	def specify_expansion(self,event):

		dialog=wx.NumberEntryDialog(self,'Enter the expansion factor that enables\nthe Detector to detect expanded areas','Enter a number:','Expansion factor',1,1,100)
		if dialog.ShowModal()==wx.ID_OK:
			self.expansion=int(dialog.GetValue())
		else:
			self.expansion=1
		self.text_expansion.SetLabel('The expansion factor is : '+str(self.expansion)+'.')
		dialog.Destroy()


	def specify_channels(self,event):

		dialog=wx.NumberEntryDialog(self,'Channel for detection','Enter a number\n(the 1st channel is 0)','Channel for detection',0,0,10)
		if dialog.ShowModal()==wx.ID_OK:
			self.detection_channel=int(dialog.GetValue())
		dialog.Destroy()

		dialog=wx.TextEntryDialog(self,'Enter the channels for analysis\n(use "," to separate each channle)','Channels for analysis')
		self.analysis_channels=[]
		if dialog.ShowModal()==wx.ID_OK:
			entry=dialog.GetValue()
			try:
				channels=entry.split(',')
				for i in channels:
					self.analysis_channels.append(int(i))
			except:
				wx.MessageBox('Please enter the number of channels for analysis in\ncorrect format! For example: 0,1,2','Error',wx.OK|wx.ICON_ERROR)
		dialog.Destroy()

		self.text_channels.SetLabel('Channel for detection: '+str(self.detection_channel)+'; Channels for analysis: '+str(self.analysis_channels))


	def analyze_cells(self,event):

		if self.path_to_files is None or self.result_path is None or self.path_to_detector is None:

			wx.MessageBox('No input file(s) / result folder / Detector.','Error',wx.OK|wx.ICON_ERROR)

		else:

			all_summary=[]
			names=[]

			for i in self.path_to_files:
				AC=AnalyzeCells(i,self.result_path,self.path_to_detector,self.cell_kinds,self.names_colors,detection_threshold=self.detection_threshold,expansion=self.expansion,show_ids=self.show_ids,
					filters=self.filters,inners=self.inners)
				AC.analyze_multichannels(detection_channel=self.detection_channel,analysis_channels=self.analysis_channels)

				basename=os.path.splitext(os.path.basename(i))[0]
				individual_path=os.path.join(self.result_path,basename)

				for cell_name in self.cell_kinds:
					individual_summary=os.path.join(individual_path,basename+'_'+cell_name+'_summary.xlsx')
					if os.path.exists(individual_summary):
						all_summary.append(pd.read_excel(individual_summary))
						names.append(basename+'_'+cell_name)

			if len(all_summary)>=1:
				all_summary=pd.concat(all_summary,keys=names,names=['File name','seq']).reset_index(level='seq',drop=True)
				all_summary.drop(all_summary.columns[0],axis=1,inplace=True)
				all_summary.to_excel(os.path.join(self.result_path,'all_summary.xlsx'),float_format='%.2f')



class WindowLv2_AnalyzeSingleChannel(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_AnalyzeSingleChannel,self).__init__(parent=None,title=title,size=(1000,340))
		self.detector_path=None
		self.path_to_detector=None
		self.cell_kinds=None
		self.path_to_files=None
		self.result_path=None
		self.detection_threshold=None
		self.expansion=None
		self.fov_dim=1280
		self.names_colors=None
		self.black_background=True
		self.show_ids=False
		self.filters={}
		
		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_inputfiles=wx.BoxSizer(wx.HORIZONTAL)
		button_inputfiles=wx.Button(panel,label='Select the TIF/SVS/JPG/PNG file(s)\nfor analyzing cells',size=(300,40))
		button_inputfiles.Bind(wx.EVT_BUTTON,self.select_files)
		wx.Button.SetToolTip(button_inputfiles,'Select one or more *.TIF or *.SVS or *.JPG or *.PNG file(s).')
		self.text_inputfiles=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_inputfiles.Add(button_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_inputfiles.Add(self.text_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe analysis results',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'Will create a subfolder for each file in the selected folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_detection=wx.BoxSizer(wx.HORIZONTAL)
		button_detection=wx.Button(panel,label='Select the Detector to\ndetect cells',size=(300,40))
		button_detection.Bind(wx.EVT_BUTTON,self.select_detector)
		wx.Button.SetToolTip(button_detection,'A trained Detector can detect cells of your interest.')
		self.text_detection=wx.StaticText(panel,label='None',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_detection.Add(button_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_detection.Add(self.text_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_detection,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_filters=wx.BoxSizer(wx.HORIZONTAL)
		button_filters=wx.Button(panel,label='Specify the filters to\nexclude unwanted cells',size=(300,40))
		button_filters.Bind(wx.EVT_BUTTON,self.specify_filters)
		wx.Button.SetToolTip(button_filters,'Select filters such as area, perimeter, roundness (1 is circle, higer value means less round), height, and width, and specify the minimum and maximum values of these filters.')
		self.text_filters=wx.StaticText(panel,label='Default: None',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_filters.Add(button_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_filters.Add(self.text_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_expansion=wx.BoxSizer(wx.HORIZONTAL)
		button_expansion=wx.Button(panel,label='Specify the expansion\nof a cell',size=(300,40))
		button_expansion.Bind(wx.EVT_BUTTON,self.specify_expansion)
		wx.Button.SetToolTip(button_expansion,'If the Detector detects the cell outlines, enter 1. If the Detector detects the nuclei, enter a number > 1 to detect areas outside the nuclei.')
		self.text_expansion=wx.StaticText(panel,label='Default: 1',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_expansion.Add(button_expansion,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_expansion.Add(self.text_expansion,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_expansion,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_analyze=wx.Button(panel,label='Start to analyze cells',size=(300,40))
		button_analyze.Bind(wx.EVT_BUTTON,self.analyze_cells)
		wx.Button.SetToolTip(button_analyze,'Will output the numbers, areas, and pixel intensities for each cell of interest.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_analyze,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_files(self,event):

		wildcard='TIF/SVS/JPG/PNG files (*.tif/*.svs/*.jpg/*.png)|*.tif;*.TIF;*.tiff;*.TIFF;*.svs;*.SVS;*.jpg;*.JPG;*.png;*.PNG'
		dialog=wx.FileDialog(self,'Select TIF/SVS/JPG/PNG file(s)','','',wildcard,style=wx.FD_MULTIPLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_files=dialog.GetPaths()
			path=os.path.dirname(self.path_to_files[0])
			self.text_inputfiles.SetLabel('Selected '+str(len(self.path_to_files))+' file(s) in: '+path+'.')
		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.result_path=dialog.GetPath()
			self.text_outputfolder.SetLabel('Results will be in: '+self.result_path+'.')
		dialog.Destroy()


	def select_detector(self,event):

		self.detector_path=os.path.join(the_absolute_current_path,'detectors')

		detectors=[i for i in os.listdir(self.detector_path) if os.path.isdir(os.path.join(self.detector_path,i))]
		if '__pycache__' in detectors:
			detectors.remove('__pycache__')
		if '__init__' in detectors:
			detectors.remove('__init__')
		if '__init__.py' in detectors:
			detectors.remove('__init__.py')
		detectors.sort()
		if 'Choose a new directory of the Detector' not in detectors:
			detectors.append('Choose a new directory of the Detector')

		dialog=wx.SingleChoiceDialog(self,message='Select a Detector',caption='Select a Detector',choices=detectors)
		if dialog.ShowModal()==wx.ID_OK:
			detector=dialog.GetStringSelection()
			if detector=='Choose a new directory of the Detector':
				dialog1=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
				if dialog1.ShowModal()==wx.ID_OK:
					self.path_to_detector=dialog1.GetPaths()
				dialog1.Destroy()
			else:
				self.path_to_detector=os.path.join(self.detector_path,detector)
			with open(os.path.join(self.path_to_detector,'model_parameters.txt')) as f:
				model_parameters=f.read()
			cell_names=json.loads(model_parameters)['cell_names']
			self.fov_dim=int(json.loads(model_parameters)['inferencing_framesize'])
			if int(json.loads(model_parameters)['black_background'])==0:
				self.black_background=True
			else:
				self.black_background=False
			if len(cell_names)>1:
				dialog1=wx.MultiChoiceDialog(self,message='Specify which cells involved in analysis',
					caption='Cell kind',choices=cell_names)
				if dialog1.ShowModal()==wx.ID_OK:
					self.cell_kinds=[cell_names[i] for i in dialog1.GetSelections()]
				else:
					self.cell_kinds=cell_names
				dialog1.Destroy()
			else:
				self.cell_kinds=cell_names
			self.names_colors={}
			self.detection_threshold={}
			colors=[str(hex_code) for hex_code in mpl.colors.cnames.values()]
			for color,cell_name in zip(colors,self.cell_kinds):
				self.names_colors[cell_name]=color
			for cell_name in self.cell_kinds:
				dialog1=ColorPicker(self,f'Color for annotating {cell_name}',[cell_name,self.names_colors[cell_name]])
				if dialog1.ShowModal()==wx.ID_OK:
					(r,b,g,_)=dialog1.color_picker.GetColour()
					new_color='#%02x%02x%02x'%(r,b,g)
					self.names_colors[cell_name]=new_color
				dialog1.Destroy()
				dialog1=wx.MessageDialog(self,'Show the IDs for each detected cells?','Show IDs?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog1.ShowModal()==wx.ID_YES:
					self.show_ids=True
				else:
					self.show_ids=False
				dialog1.Destroy()
				dialog1=wx.NumberEntryDialog(self,'Detection threshold for '+str(cell_name),'Enter an number between 0 and 100','Detection threshold for '+str(cell_name),0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					self.detection_threshold[cell_name]=int(dialog1.GetValue())/100
				else:
					self.detection_threshold[cell_name]=0
				dialog1.Destroy()
			self.text_detection.SetLabel('Detector: '+detector+'; '+'The cell kinds / detection threshold: '+str(self.detection_threshold)+'.')
		dialog.Destroy()


	def specify_filters(self,event):

		filters_choices=['area','perimeter','roundness','height','width']

		dialog=wx.MultiChoiceDialog(self,message='Select filters to exclude unwanted cells',caption='Filters',choices=filters_choices)
		if dialog.ShowModal()==wx.ID_OK:
			selected_filters=[filters_choices[i] for i in dialog.GetSelections()]
		else:
			selected_filters=[]
		dialog.Destroy()

		for ft in selected_filters:
			dialog=wx.NumberEntryDialog(self,'The min value for '+str(ft),'The unit is pixel (except for roundness)','The min value for '+str(ft),0,0,100000000000000)
			values=[0,np.inf]
			if dialog.ShowModal()==wx.ID_OK:
				values[0]=int(dialog.GetValue())
			dialog.Destroy()
			dialog=wx.NumberEntryDialog(self,'The max value (enter 0 for infinity) for '+str(ft),'The unit is pixel (except for roundness)','The max value for '+str(ft),0,0,100000000000000)
			if dialog.ShowModal()==wx.ID_OK:
				value=int(dialog.GetValue())
				if value>0:
					values[1]=value
			dialog.Destroy()
			self.filters[ft]=values

		if len(self.filters)>0:
			self.self.text_filters.SetLabel('Filters: '+str(self.filters))
		else:
			self.self.text_filters.SetLabel('NO filters selected.')


	def specify_expansion(self,event):

		dialog=wx.NumberEntryDialog(self,'Enter the expansion factor that enables\nthe Detector to detect expanded areas','Enter a number:','Expansion factor',1,1,100)
		if dialog.ShowModal()==wx.ID_OK:
			self.expansion=int(dialog.GetValue())
		else:
			self.expansion=1
		self.text_expansion.SetLabel('The expansion factor is : '+str(self.expansion)+'.')
		dialog.Destroy()


	def analyze_cells(self,event):

		if self.path_to_files is None or self.result_path is None or self.path_to_detector is None:

			wx.MessageBox('No input file(s) / result folder / Detector.','Error',wx.OK|wx.ICON_ERROR)

		else:

			all_summary=[]
			names_summary=[]
			all_arearatios=[]
			names_arearatios=[]

			for i in self.path_to_files:
				AC=AnalyzeCells(i,self.result_path,self.path_to_detector,self.cell_kinds,self.names_colors,detection_threshold=self.detection_threshold,expansion=self.expansion,show_ids=self.show_ids,
					filters=self.filters,inners=self.inners)
				AC.analyze_singlechannel()

				basename=os.path.splitext(os.path.basename(i))[0]
				individual_path=os.path.join(self.result_path,basename)

				for cell_name in self.cell_kinds:
					individual_summary=os.path.join(individual_path,basename+'_'+cell_name+'_summary.xlsx')
					individual_arearatio=os.path.join(individual_path,basename+'_'+cell_name+'_arearatio.xlsx')
					if os.path.exists(individual_summary):
						all_summary.append(pd.read_excel(individual_summary))
						names_summary.append(basename+'_'+cell_name)
					if os.path.exists(individual_arearatio):
						all_arearatios.append(pd.read_excel(individual_arearatio))
						names_arearatios.append(basename+'_'+cell_name)

			if len(all_summary)>=1:
				all_summary=pd.concat(all_summary,keys=names_summary,names=['File name','seq']).reset_index(level='seq',drop=True)
				all_summary.drop(all_summary.columns[0],axis=1,inplace=True)
				all_summary.to_excel(os.path.join(self.result_path,'all_summary.xlsx'),float_format='%.2f')
			if len(all_arearatios)>=1:
				all_arearatios=pd.concat(all_arearatios,keys=names_arearatios,names=['File name','seq']).reset_index(level='seq',drop=True)
				all_arearatios.drop(all_arearatios.columns[0],axis=1,inplace=True)
				all_arearatios.to_excel(os.path.join(self.result_path,'all_arearatios.xlsx'),float_format='%.6f')



class WindowLv2_CalculateTotalIntensity(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_CalculateTotalIntensity,self).__init__(parent=None,title=title,size=(1000,200))
		self.path_to_files=None
		self.result_path=None
		
		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_inputfiles=wx.BoxSizer(wx.HORIZONTAL)
		button_inputfiles=wx.Button(panel,label='Select the LIF/TIF/QPTIFF file(s)\nfor analyzing cells',size=(300,40))
		button_inputfiles.Bind(wx.EVT_BUTTON,self.select_files)
		wx.Button.SetToolTip(button_inputfiles,'Select one or more *.LIF or *.TIF or *.QPTIFF file(s).')
		self.text_inputfiles=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_inputfiles.Add(button_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_inputfiles.Add(self.text_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_inputfiles,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe analysis results',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'Will create a subfolder for each file in the selected folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_analyze=wx.Button(panel,label='Start to calculate intensity',size=(300,40))
		button_analyze.Bind(wx.EVT_BUTTON,self.calculate_intensity)
		wx.Button.SetToolTip(button_analyze,'Will calculate total pixel intensities for each channel of an image.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_analyze,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_files(self,event):

		wildcard='LIF/TIF/QPTIFF files (*.lif/*.tif/*.qptiff)|*.lif;*.LIF;*.tif;*.TIF;*.tiff;*.TIFF;*.qptiff;*.QPTIFF'
		dialog=wx.FileDialog(self,'Select LIF/TIF/QPTIFF file(s)','','',wildcard,style=wx.FD_MULTIPLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_files=dialog.GetPaths()
			path=os.path.dirname(self.path_to_files[0])
			self.text_inputfiles.SetLabel('Selected '+str(len(self.path_to_files))+' file(s) in: '+path+'.')
		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.result_path=dialog.GetPath()
			self.text_outputfolder.SetLabel('Results will be in: '+self.result_path+'.')
		dialog.Destroy()


	def calculate_intensity(self,event):

		if self.path_to_files is None or self.result_path is None:

			wx.MessageBox('No input file(s) / result folder.','Error',wx.OK|wx.ICON_ERROR)

		else:

			all_intensities=[]
			names=[]

			for i in self.path_to_files:

				calculate_totalintensity(i,self.result_path)

				basename=os.path.splitext(os.path.basename(i))[0]
				individual_path=os.path.join(self.result_path,basename)

				individual_intensity=os.path.join(individual_path,basename+'_total_intensity.xlsx')
				if os.path.exists(individual_intensity):
					all_intensities.append(pd.read_excel(individual_intensity))
					names.append(basename)

			if len(all_intensities)>=1:
				all_intensities=pd.concat(all_intensities,keys=names,names=['File name','seq']).reset_index(level='seq',drop=True)
				all_intensities.drop(all_intensities.columns[0],axis=1,inplace=True)
				all_intensities.to_excel(os.path.join(self.result_path,'all_intensities.xlsx'),float_format='%.2f')



def main_window():

	the_absolute_current_path=str(Path(__file__).resolve().parent)
	app=wx.App()
	InitialWindow(f'Cellan v{__version__}')
	print('The user interface initialized!')
	app.MainLoop()


if __name__=='__main__':

	main_window()


