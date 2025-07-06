from tkinter import *
from tkinter import filedialog, ttk
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Flatten, Conv2D, Dropout
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, RMSprop, Adadelta
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras_tuner import RandomSearch, Hyperband
import warnings
import logging
from model.data_utils import load_dynamic_dataset

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

# Import custom modules
from model.data_utils import load_or_generate_dataset
from model.cnn_model import build_cnn_model, train_model
from model.gradcam import gradcam_heatmap
from model.metrics import calculate_metrics, plot_confusion_matrix, plot_training_history, plot_comparison_graph

# Initialize the main window
main = Tk()
main.title("Comparative Study of Breast Cancer Using Machine Learning")
main.geometry("1500x1200")

# Global variables
global filename, x, y, X_train, X_test, y_train, y_test, adam_cnn, rmsprop_cnn, adadelta_cnn, type_counts
filename = None
X = None
Y = None
X_train = None
X_test = None
y_train = None
y_test = None
adam_cnn = None
rmsprop_cnn = None
adadelta_cnn = None
type_counts = None

adam_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": [], "specificity": []}
rmsprop_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": [], "specificity": []}
adadelta_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": [], "specificity": []}
labels = ['Type-D', 'Type-L', 'Type-M']
label_mapping = {'Type-D': 'Ductal Carcinoma', 'Type-L': 'Lobular Carcinoma', 'Type-M': 'Mucinous Carcinoma'}

# Dynamic parameters
img_size = IntVar(value=64)
lr_var = DoubleVar(value=0.001)
epoch_var = IntVar(value=30)
classification_type_var = StringVar(value='classificacao_binaria')
magnification_var = StringVar(value='40X')

classification_types = ['classificacao_binaria', 'classificacao_multiclasse']
magnifications = ['40X', '100X', '200X', '400X']

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    if not filename or not os.path.exists(filename):
        text.delete('1.0', END)
        text.insert(END, "Please select a valid directory\n")
        filename = None
    else:
        text.delete('1.0', END)
        text.insert(END, f"Dataset loaded from: {filename}\n")

def processDataset():
    global X, Y, X_train, X_test, y_train, y_test, type_counts, labels
    text.delete('1.0', END)
    if filename is None:
        text.insert(END, "Please upload a valid dataset first.\n")
        return
    try:
        text.insert(END, f"Processing dataset from: {filename}\n")
        text.insert(END, f"Classification type: {classification_type_var.get()}\n")
        text.insert(END, f"Magnification: {magnification_var.get()}\n")
        text.insert(END, f"Image size: {img_size.get()}x{img_size.get()}\n\n")
        main.update()
        # Use the new dynamic loader
        X, Y, class_names = load_dynamic_dataset(
            filename,
            classification_type_var.get(),
            magnification_var.get(),
            img_size.get()
        )
        if X is None or Y is None or len(X) == 0 or len(Y) == 0:
            text.insert(END, "Error: No data was loaded. Please check your folder structure and selections.\n")
            text.insert(END, f"Current directory contents: {os.listdir(filename)}\n")
            return
        labels = class_names
        text.insert(END, f"Successfully loaded {len(X)} images\n")
        main.update()
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, Y = X[indices], Y[indices]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        class_counts = np.bincount(np.argmax(Y, axis=1))
        type_counts = dict(zip(labels, class_counts))
        text.insert(END, f"Total number of images found in dataset: {len(X)}\n")
        text.insert(END, f"Total cancer labels found in dataset: {labels}\n")
        text.insert(END, "80% images used for training and 20% for testing\n\n")
        text.insert(END, f"Training Images Size = {X_train.shape[0]}\n")
        text.insert(END, f"Testing Images Size = {X_test.shape[0]}\n\n")
        names, count = np.unique(np.argmax(Y, axis=1), return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(labels)), count)
        plt.xticks(np.arange(len(labels)), labels)
        plt.title("Images Count Found in Dataset")
        plt.xlabel("Cancer Type")
        plt.ylabel("Count")
        for i, v in enumerate(count):
            plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        plt.show()
    except Exception as e:
        text.insert(END, f"Error processing dataset: {str(e)}\n")
        text.insert(END, "Please make sure the dataset directory structure is correct.\n")

def getAccuracy(algorithm, predict, y_test, metrics_dict):
    metrics, cm = calculate_metrics(y_test, predict)
    
    for key in metrics:
        metrics_dict[key].append(metrics[key])
    
    text.insert(END, f"{algorithm} Accuracy    :  {metrics['accuracy']}\n")
    text.insert(END, f"{algorithm} Precision   : {metrics['precision']}\n")
    text.insert(END, f"{algorithm} Sensitivity : {metrics['recall']}\n")
    text.insert(END, f"{algorithm} F1-Score    : {metrics['f1_score']}\n")
    text.insert(END, f"{algorithm} Specificity : {metrics['specificity']}\n\n")
    
    plot_confusion_matrix(cm, labels, f"{algorithm} Confusion matrix")

def trainAdam():
    global adam_cnn
    if X_train is None or y_train is None:
        text.insert(END, "Please preprocess the dataset first!\n")
        return
    adam_cnn = build_cnn_model(X_train.shape[1:], y_train.shape[1])
    history = train_model(
        adam_cnn, X_train, y_train, X_test, y_test,
        "Adam", lr_var.get(), epoch_var.get(),
        "model/adam_cnn_weights.keras", "model/adam_history.pckl",
        text, main
    )
    if history:
        plot_training_history(history, "Adam")
    predict = np.argmax(adam_cnn.predict(X_test), axis=1)
    ytest = np.argmax(y_test, axis=1)
    getAccuracy("Adam", predict, ytest, adam_metrics)

def trainRMSprop():
    global rmsprop_cnn
    if X_train is None or y_train is None:
        text.insert(END, "Please preprocess the dataset first!\n")
        return
    rmsprop_cnn = build_cnn_model(X_train.shape[1:], y_train.shape[1])
    history = train_model(
        rmsprop_cnn, X_train, y_train, X_test, y_test,
        "RMSprop", lr_var.get(), epoch_var.get(),
        "model/rmsprop_cnn_weights.keras", "model/rmsprop_history.pckl",
        text, main
    )
    if history:
        plot_training_history(history, "RMSprop")
    predict = np.argmax(rmsprop_cnn.predict(X_test), axis=1)
    ytest = np.argmax(y_test, axis=1)
    getAccuracy("RMSprop", predict, ytest, rmsprop_metrics)

def trainAdadelta():
    global adadelta_cnn
    if X_train is None or y_train is None:
        text.insert(END, "Please preprocess the dataset first!\n")
        return
    adadelta_cnn = build_cnn_model(X_train.shape[1:], y_train.shape[1])
    history = train_model(
        adadelta_cnn, X_train, y_train, X_test, y_test,
        "Adadelta", lr_var.get(), epoch_var.get(),
        "model/adadelta_cnn_weights.keras", "model/adadelta_history.pckl",
        text, main
    )
    if history:
        plot_training_history(history, "Adadelta")
    predict = np.argmax(adadelta_cnn.predict(X_test), axis=1)
    ytest = np.argmax(y_test, axis=1)
    getAccuracy("Adadelta", predict, ytest, adadelta_metrics)

def get_best_model():
    # Only consider models that are not None
    candidates = []
    if adam_cnn is not None and len(adam_metrics['accuracy']) > 0:
        candidates.append((adam_cnn, np.mean(adam_metrics['accuracy'])))
    if rmsprop_cnn is not None and len(rmsprop_metrics['accuracy']) > 0:
        candidates.append((rmsprop_cnn, np.mean(rmsprop_metrics['accuracy'])))
    if adadelta_cnn is not None and len(adadelta_metrics['accuracy']) > 0:
        candidates.append((adadelta_cnn, np.mean(adadelta_metrics['accuracy'])))
    if not candidates:
        return None
    # Return the model with the highest accuracy
    return max(candidates, key=lambda x: x[1])[0]

def showCancerRegion():
    file_path = filedialog.askopenfilename(initialdir=".", title="Select Image", filetypes=(("png files", ".png"),))
    if file_path:
        best_model = get_best_model()
        if best_model is None:
            text.insert(END, "No trained model available. Please train a model first!\n")
            return
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (img_size.get(), img_size.get())).astype('float32') / 255.0
        image_expanded = np.expand_dims(image_resized, axis=0)
        
        # Ensure the model is built by calling it once if not already built
        if not hasattr(best_model, 'inputs') or best_model.inputs is None:
            best_model.predict(image_expanded)
        
        cam = gradcam_heatmap(best_model, image_expanded, 'conv2d_2')
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        prediction = best_model.predict(image_expanded)
        predicted_class_index = np.argmax(prediction)
        predicted_class = labels[predicted_class_index]
        user_friendly_label = label_mapping.get(predicted_class, predicted_class)
        threshold = 0.5
        cancer_region_percentage = np.sum(cam > threshold) / cam.size * 100
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        axs[0].imshow(image)
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        axs[1].imshow(image)
        axs[1].imshow(cam, cmap='jet', alpha=0.5)
        axs[1].set_title(f'Predicted: {user_friendly_label}\nCancer Region: {cancer_region_percentage:.2f}%')
        axs[1].axis('off')
        plt.show()

def graph():
    plot_comparison_graph(
        (adam_metrics, "Adam"),
        (rmsprop_metrics, "RMSprop"),
        (adadelta_metrics, "Adadelta")
    )

def exit():
    main.destroy()

# Create gradient background
def create_gradient(canvas, width, height, color1, color2):
    gradient = PhotoImage(width=width, height=height)
    line_color = lambda i: f'#{int(color1[1:3], 16) + (int(color2[1:3], 16) - int(color1[1:3], 16)) * i // width:02x}{int(color1[3:5], 16) + (int(color2[3:5], 16) - int(color1[3:5], 16)) * i // width:02x}{int(color1[5:7], 16) + (int(color2[5:7], 16) - int(color1[5:7], 16)) * i // width:02x}'
    for x in range(width):
        gradient.put(line_color(x), to=(x, 0, x + 1, height))
    canvas.create_image(0, 0, image=gradient, anchor="nw")
    canvas.gradient = gradient

# Create UI elements
font = ('times', 16, 'bold')
title = Label(main, text='Comparative Study Of Breast Cancer Using Machine Learning', bg='black', fg='white', font=font, height=3, width=120)
title.place(x=0, y=6)

canvas = Canvas(main, bg="white", height=200, width=1500)
canvas.place(x=0, y=70)

create_gradient(canvas, 1500, 200, "#A9A9A9", "#D3D3D3")  

# Add dynamic parameter controls
ttk.Label(main, text="Image Size:").place(x=1050, y=100)
size_dropdown = ttk.Combobox(main, textvariable=img_size, values=[32, 64, 128, 224], width=5)
size_dropdown.place(x=1130, y=100)

ttk.Label(main, text="LR:").place(x=1050, y=130)
ttk.Entry(main, textvariable=lr_var, width=6).place(x=1085, y=130)
ttk.Label(main, text="Epochs:").place(x=1130, y=130)
ttk.Entry(main, textvariable=epoch_var, width=6).place(x=1190, y=130)

ttk.Label(main, text="Classification Type:").place(x=900, y=100)
classification_dropdown = ttk.Combobox(main, textvariable=classification_type_var, values=classification_types, width=20)
classification_dropdown.place(x=1040, y=100)

ttk.Label(main, text="Magnification:").place(x=900, y=130)
magnification_dropdown = ttk.Combobox(main, textvariable=magnification_var, values=magnifications, width=8)
magnification_dropdown.place(x=1040, y=130)

# Create buttons
font1 = ('times', 14, 'bold')
button1 = Button(main, text="Upload Cancer Dataset", command=upload, font=font1)
button1.place(x=50, y=100)
button2 = Button(main, text="Preprocess Dataset", command=processDataset, font=font1)
button2.place(x=300, y=100)
button3 = Button(main, text="Train Adam Optimizer", command=trainAdam, font=font1)
button3.place(x=550, y=100)
button4 = Button(main, text="Train RMSprop Optimizer", command=trainRMSprop, font=font1)
button4.place(x=800, y=100)
button5 = Button(main, text="Train Adadelta Optimizer", command=trainAdadelta, font=font1)
button5.place(x=50, y=160)
button6 = Button(main, text="Show Comparison Graph", command=graph, font=font1)
button6.place(x=300, y=160)
button7 = Button(main, text="Show Cancer Region", command=showCancerRegion, font=font1)
button7.place(x=550, y=160)
button8 = Button(main, text="Exit", command=exit, font=font1)
button8.place(x=800, y=160)

# Add hover effect to buttons
def on_enter(e):
    e.widget['background'] = '#555555'
def on_leave(e):
    e.widget['background'] = 'SystemButtonFace'

for button in [button1, button2, button3, button4, button5, button6, button7, button8]:
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

# Create text area
font1 = ('times', 12, 'bold')
text = Text(main, height=25, width=175, font=font1)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=300)

main.config(bg='grey')
main.mainloop()
