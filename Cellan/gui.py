import os
import cv2
import wx
import wx.lib.agw.hyperlink as hl
from pathlib import Path
import json
import shutil
from .analyzer import AnalyzeCells
from .detector import Detector
from .tools import extract_images
from Cellan import __version__



the_absolute_current_path=str(Path(__file__).resolve().parent)



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
		button_train=wx.Button(panel,label='Training Module',size=(200,40))
		button_train.Bind(wx.EVT_BUTTON,self.window_train)
		wx.Button.SetToolTip(button_train,'Teach Cellan to recognize the cells of your interest.')
		button_analyze=wx.Button(panel,label='Analysis Module',size=(200,40))
		button_analyze.Bind(wx.EVT_BUTTON,self.window_analyze)
		wx.Button.SetToolTip(button_analyze,'Use Cellan to analyze cells.')
		module_modules.Add(button_train,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_modules.Add(button_analyze,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_modules,0,wx.ALIGN_CENTER,50)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def window_train(self,event):

		WindowLv1_TrainingModule('Training Module')


	def window_analyze(self,event):

		WindowLv1_AnalysisModule('Analysis Module')



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
			'Extract images from LIF/TIF files for annotation of the cells of your interest. See Extended Guide for how to select images to annotate.')
		boxsizer.Add(button_generateimages,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		link_annotate=wx.lib.agw.hyperlink.HyperLinkCtrl(panel,0,'\nAnnotate images with Roboflow\n',URL='https://roboflow.com')
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

		super(WindowLv1_AnalysisModule,self).__init__(parent=None,title=title,size=(500,170))
		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,40,0)

		button_analyzeintensity=wx.Button(panel,label='Analyze Signal Intensity',size=(300,40))
		button_analyzeintensity.Bind(wx.EVT_BUTTON,self.analyze_intensity)
		wx.Button.SetToolTip(button_analyzeintensity,
			'Automatically detect cells of your interest and analyze the pixel intensities in them.')
		boxsizer.Add(button_analyzeintensity,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,30,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def analyze_intensity(self,event):

		WindowLv2_AnalyzeIntensity('Analyze Signal Intensity')



class WindowLv2_GenerateImages(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_GenerateImages,self).__init__(parent=None,title=title,size=(1000,240))
		self.path_to_files=None
		self.result_path=None
		self.fov_div=1

		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_inputfiles=wx.BoxSizer(wx.HORIZONTAL)
		button_inputfiles=wx.Button(panel,label='Select the *.LIF/*.TIF file(s) to generate\nimage examples',size=(300,40))
		button_inputfiles.Bind(wx.EVT_BUTTON,self.select_files)
		wx.Button.SetToolTip(button_inputfiles,'Select one or more *.LIF/*.TIF files.')
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
		button_fov=wx.Button(panel,label='Specify the field of view\nin an image',size=(300,40))
		button_fov.Bind(wx.EVT_BUTTON,self.specify_fov)
		wx.Button.SetToolTip(button_fov,'Specify the number (n) of field of view for height/width, the image will be divided into smaller field of view with the dimension of (height/n) X (width/n).')
		self.text_fov=wx.StaticText(panel,label='Default: 1',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
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

		wildcard='LIF/TIF files (*.lif/*.tif)|*.lif;*.LIF;*.tif;*.TIF;*.tiff;*.TIFF'
		dialog=wx.FileDialog(self,'Select LIF/TIF file(s)','','',wildcard,style=wx.FD_MULTIPLE)
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

		dialog=wx.NumberEntryDialog(self,'Enter the number of sections\nthe width and height should be divided','Enter a number:','Number of field of view',1,1,10000)
		if dialog.ShowModal()==wx.ID_OK:
			self.fov_div=int(dialog.GetValue())
		else:
			self.fov_div=1
		self.text_fov.SetLabel('The height and width of an image will be divided by : '+str(self.fov_div)+'.')
		dialog.Destroy()


	def generate_images(self,event):

		if self.path_to_files is None or self.result_path is None:

			wx.MessageBox('No input file(s) / output folder selected.','Error',wx.OK|wx.ICON_ERROR)

		else:

			print('Generating image examples...')
			for i in self.path_to_files:
				extract_images(i,self.result_path,self.fov_div)
			print('Image example generation completed!')



class WindowLv2_TrainDetectors(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_TrainDetectors,self).__init__(parent=None,title=title,size=(1000,280))
		self.path_to_trainingimages=None
		self.path_to_annotation=None
		self.inference_size=1280
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

		module_inferencingsize=wx.BoxSizer(wx.HORIZONTAL)
		button_inferencingsize=wx.Button(panel,label='Specify the inferencing framesize\nfor the Detector to train',size=(300,40))
		button_inferencingsize.Bind(wx.EVT_BUTTON,self.input_inferencingsize)
		wx.Button.SetToolTip(button_inferencingsize,'Should be an even number. Larger size means higher accuracy but slower speed.')
		self.text_inferencingsize=wx.StaticText(panel,label='Default: 1280.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_inferencingsize.Add(button_inferencingsize,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_inferencingsize.Add(self.text_inferencingsize,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_inferencingsize,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
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


	def input_inferencingsize(self,event):

		dialog=wx.NumberEntryDialog(self,'Input the inferencing frame size\nof the Detector to train','Enter a number:','Divisible by 2',1280,1,2048)
		if dialog.ShowModal()==wx.ID_OK:
			self.inference_size=int(dialog.GetValue())
			self.text_inferencingsize.SetLabel('Inferencing frame size: '+str(self.inference_size)+'.')
		dialog.Destroy()
		

	def input_iterations(self,event):

		dialog=wx.NumberEntryDialog(self,'Input the iteration number\nfor the Detector training','Enter a number:','Iterations',5000,1,1000000)
		if dialog.ShowModal()==wx.ID_OK:
			self.iteration_num=int(dialog.GetValue())
			self.text_iterations.SetLabel('Training iteration number: '+str(self.iteration_num)+'.')
		dialog.Destroy()


	def train_detector(self,event):

		if self.path_to_trainingimages is None or self.path_to_annotation is None:

			wx.MessageBox('No training images / annotation file selected.','Error',wx.OK|wx.ICON_ERROR)

		else:

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
				DT.train(self.path_to_annotation,self.path_to_trainingimages,self.path_to_detector,self.iteration_num,self.inference_size)



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



class WindowLv2_AnalyzeIntensity(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_AnalyzeIntensity,self).__init__(parent=None,title=title,size=(1000,380))
		self.detector_path=None
		self.path_to_detector=None
		self.cell_kinds=None
		self.path_to_files=None
		self.result_path=None
		self.detection_threshold=None
		self.expansion=None
		self.fov_div=1
		self.names_colors=None
		self.detection_channel=0
		self.analysis_channels=[]
		
		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_inputfiles=wx.BoxSizer(wx.HORIZONTAL)
		button_inputfiles=wx.Button(panel,label='Select the *.LIF/*.TIF file(s)\nfor analyzing cells',size=(300,40))
		button_inputfiles.Bind(wx.EVT_BUTTON,self.select_files)
		wx.Button.SetToolTip(button_inputfiles,'Select one or more *.LIF/*.TIF file(s).')
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

		module_fov=wx.BoxSizer(wx.HORIZONTAL)
		button_fov=wx.Button(panel,label='Specify the field of view\nin an image',size=(300,40))
		button_fov.Bind(wx.EVT_BUTTON,self.specify_fov)
		wx.Button.SetToolTip(button_fov,'Specify the number (n) of field of view for height/width, the image will be divided into smaller field of view with the dimension of (height/n) X (width/n).')
		self.text_fov=wx.StaticText(panel,label='Default: 1',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_fov.Add(button_fov,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_fov.Add(self.text_fov,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_fov,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
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
		button_analyze.Bind(wx.EVT_BUTTON,self.analyze_intensity)
		wx.Button.SetToolTip(button_analyze,'Will output ...')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_analyze,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_files(self,event):

		wildcard='LIF/TIF files (*.lif/*.tif)|*.lif;*.LIF;*.tif;*.TIF;*.tiff;*.TIFF'
		dialog=wx.FileDialog(self,'Select LIF/TIF file(s)','','',wildcard,style=wx.FD_MULTIPLE)
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
			if len(self.cell_kinds)>1:
				diff=int(255/len(self.cell_kinds))
			else:
				diff=0
			for cell_name in self.cell_kinds:
				dialog1=wx.NumberEntryDialog(self,'Detection threshold for '+str(cell_name),'Enter an number between 0 and 100','Detection threshold for '+str(cell_name),0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					self.detection_threshold[cell_name]=int(dialog1.GetValue())/100
				else:
					self.detection_threshold[cell_name]=0
				dialog1.Destroy()
				self.names_colors[cell_name]=(255,255-diff,255)
			self.text_detection.SetLabel('Detector: '+detector+'; '+'The cell kinds / detection threshold: '+str(self.detection_threshold)+'.')
		dialog.Destroy()


	def specify_fov(self,event):

		dialog=wx.NumberEntryDialog(self,'Enter the number of sections\nthe width and height should be divided','Enter a number:','Number of field of view',1,1,10000)
		if dialog.ShowModal()==wx.ID_OK:
			self.fov_div=int(dialog.GetValue())
		else:
			self.fov_div=1
		self.text_fov.SetLabel('The height and width of an image will be divided by : '+str(self.fov_div)+'.')
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


	def analyze_intensity(self,event):

		if self.path_to_files is None or self.result_path is None or self.path_to_detector is None:

			wx.MessageBox('No input file(s) / result folder / Detector.','Error',wx.OK|wx.ICON_ERROR)

		else:

			for i in self.path_to_files:
				AC=AnalyzeCells(i,self.result_path,self.path_to_detector,self.cell_kinds,detection_threshold=self.detection_threshold,expansion=self.expansion,fov_div=self.fov_div)
				AC.channels_intensity(self.names_colors,detection_channel=self.detection_channel,analysis_channels=self.analysis_channels)



def main_window():

	the_absolute_current_path=str(Path(__file__).resolve().parent)
	app=wx.App()
	InitialWindow(f'Cellan v{__version__}')
	print('The user interface initialized!')
	app.MainLoop()


if __name__=='__main__':

	main_window()


