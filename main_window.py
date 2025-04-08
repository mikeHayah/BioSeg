import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QThread
from checkerQt import Checker
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QCheckBox, QComboBox, QLabel, QSlider, QScrollArea)
from PyQt5.QtCore import Qt
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from predict_patched import make_predictions
from infer_patches import *

class MainWindow(QWidget):
    def __init__(self, solver, config):
        super().__init__()
        self.mode = config.mode
        self.test_path = config.test_path
        self.solver = solver
        self.checker = Checker(self.solver)

        self.solver.set_checker(self.checker)

        # Layout and Widgets
        layout = QVBoxLayout()

        # Main layout
        layout = QVBoxLayout()

        # Checkbox to enable/disable hook
        self.toggle_checkbox = QCheckBox("Enable Hook")
        self.toggle_checkbox.stateChanged.connect(self.toggle_hook)
        layout.addWidget(self.toggle_checkbox)

        # Toggle Button (QCheckBox styled as a slider)
        self.visual_mode = self.checker.visualize_samples
        self.toggle = QCheckBox("Mode: Visualize samples", self)
        self.toggle.setCheckState(Qt.Checked)
        self.toggle.setStyleSheet("""
            QCheckBox {
                spacing: 10px;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 50px;
                height: 25px;
                border-radius: 12px;
                border: 2px solid #555;
                background: #ccc;
            }
            QCheckBox::indicator:checked {
                background: #66ccff;
                border: 2px solid #0055aa;
            }
        """)
        self.toggle.stateChanged.connect(self.toggle_mode)
        layout.addWidget(self.toggle)

        # ComboBox to select layer for hook
        self.layer_selector = QComboBox()
        self.populate_layer_selector()
        self.layer_selector.currentIndexChanged.connect(self.update_hook_layer)
        layout.addWidget(QLabel("Select Layer:"))
        layout.addWidget(self.layer_selector)

        # Checkbox to enable/disable hook for gradient
        self.toggle_grad = QCheckBox("Enable Gradient Hook")
        self.toggle_grad.stateChanged.connect(self.grad_registerion)
        layout.addWidget(self.toggle_grad)

        # Label inside the Scroll Area to display the grad value
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMaximumHeight(100)
        layout.addWidget(self.scroll_area)
        self.value_label = QLabel("Gradient: None\n" , self)
        self.value_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.value_label.setWordWrap(True)  # Enable word wrap for text
        self.scroll_area.setWidget(self.value_label)


        # Visualization canvas
        self.checker.canvas = FigureCanvas(self.checker.canvas.figure)
        layout.addWidget(self.checker.canvas)

        # Visualization canvas for training loss and validation loss
        self.solver.canvas = FigureCanvas(self.solver.canvas.figure)
        layout.addWidget(self.solver.canvas)

        # Label to dispay minimum, maximum, mean, std values of the layer
        self.min_label = QLabel("Min: None", self)
        self.max_label = QLabel("Max: None", self)
        self.mean_label = QLabel("Mean: None", self)
        self.std_label = QLabel("Std: None", self)
        layout.addWidget(self.min_label)
        layout.addWidget(self.max_label)
        layout.addWidget(self.mean_label)
        layout.addWidget(self.std_label)

        # Button to play and pause visualization update
        self.was_play = False
        self.update_button = QPushButton("Play")
        self.update_button.clicked.connect(self.play_pause)
        layout.addWidget(self.update_button)

        # Button to trigger visualization update
        self.visual_sample = True
        self.one_shot = QPushButton("One Shot")
        self.one_shot.clicked.connect(self.take_shot)
        layout.addWidget(self.one_shot)

        self.setLayout(layout)
        self.setWindowTitle("Model Trainer and Checker")

        # Thread for training
        self.train_thread = QThread()
        self.solver.moveToThread(self.train_thread)
        self.train_thread.started.connect(self.solver.train)
        self.solver.finished.connect(self.on_training_finished)
        self.solver.finished.connect(self.train_thread.quit)

        # Thread for inference
        self.inference_thread = QThread()
        
        
       
    def toggle_hook(self, state):
        self.checker.flag = state == 2  # Checkbox state: 0=unchecked, 2=checked
        self.checker.update_hook()

    def populate_layer_selector(self):
        if self.solver.model is not None:
            for name, layer in self.solver.model.named_modules():
                if isinstance(layer, (torch.nn.modules.Conv2d, torch.nn.modules.ConvTranspose2d)):
                    self.layer_selector.addItem(name)
                    self.checker.hook_layer_index = name
            #self.layer_selector.setCurrentIndex(0)
            self.layer_selector.setCurrentIndex(self.layer_selector.count() - 1)
                
    def update_hook_layer(self, index):
        layer_name = self.layer_selector.itemText(index)
        self.checker.set_hook_layer(layer_name)

    def toggle_mode(self, state):
        try:
            self.solver.update_signal.disconnect(self.visual_mode)
        except TypeError:
            pass 
        if state == Qt.Checked:
            self.visual_sample = True
            self.toggle.setText("Mode: Visualize samples")  
            self.visual_mode = self.checker.visualize_samples
        else:
            self.visual_sample = False
            self.toggle.setText("Mode: Visualize channels")  
            self.visual_mode = self.checker.visualize_channels
        if self.was_play:
            self.solver.update_signal.connect(self.visual_mode)
     
    def grad_registerion(self, state):
        if state == 2:
            self.checker.register_for_grad()
        else:
            self.checker.deregister_for_grad()
    
    def play_pause(self):
        if (self.was_play):
            #self.solver.update_signal.disconnect(self.checker.update_visualization)
            self.solver.update_signal.disconnect(self.visual_mode)
            self.was_play = False
            self.update_button.setText("Play")
            self.layer_selector.setEnabled(True)
            self.toggle_checkbox.setEnabled(True)
        else:
            # Connect signal from solver to checker
            #self.solver.update_signal.connect(self.checker.update_visualization)
            self.solver.update_signal.connect(self.visual_mode)
            self.was_play = True
            self.update_button.setText("Pause")
            self.layer_selector.setEnabled(False)
            self.toggle_checkbox.setEnabled(False)


    def take_shot(self):
        if self.visual_sample:
            self.checker.visualize_samples()       
        else:
            self.checker.visualize_channels()      
        grad_value = self.checker.get_grad() 
        self.value_label.setText(f"Grad Value: {grad_value}")
        data_min, data_max, data_mean, data_std = self.checker.get_statis()
        self.min_label.setText(f"Min: {data_min}")
        self.max_label.setText(f"Max: {data_max}")
        self.mean_label.setText(f"Mean: {data_mean}")
        self.std_label.setText(f"Std: {data_std}")


            

    def start(self):
        if (self.mode == 'train'):
            self.train_thread.start()
        else:
            self.inference_thread.started.connect(make_prediction("./models/networkOm3/U_Net_plus-94-10.5900-2.9149-best_train.pkl", self.test_path))
            #self.inference_thread.started.connect(normalize_patches (self.test_path))
            #self.inference_thread.started.connect(post3d_process(self.test_path))
            #self.inference_thread.started.connect(img_mode(self.test_path))
            #self.inference_thread.started.connect(hugging_face_inference(self.test_path))
            #self.inference_thread.start()

    def on_training_finished(self):
        print("Training finished!")