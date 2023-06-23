# drowsiness_detection_cnn
Project for our 3rd-year minor defense created by me and my friend Kapil Kunwar ( https://github.com/undef125 )

# Dataset
1) For eyes: the dataset is custom
2) For yawn: the dataset is a mixture of images from Kaggle collected by us

# Processing dataset
1) image are gathered and the ROI is cropped from images and then split manually for test and train
2) Then data augmentation is performed to increase the number of images

# Model
CNN model as:
1) Convolution layer
    It consists of 3 2D convolution layers and used MaxPooling
2) Flatten layer:
    Output from the convolution layer is fed to flatten layer
3) Fully connected layer:
    It consists of 3 layers. The first one is the input layer where the output from flatten layer is fed
    The second one is the hidden layer and the last one output layer as sigmoid as an activation function
    which gives results between 0 and 1
    
# How To Run
1) Git clone repo using
    ```sh 
    https://github.com/undef125/drowsiness_detection_cnn.git
    ```
2) Change the directory to our project:
    ```sh
    cd drowsiness_detection_cnn
    ```
4) Install the requirements using:
    ```sh
    pip install -r requirements.txt
    ```
5) For GUI:
    ```sh
    python drowsi_gui.py
    ```
    Or
    For terminal based:
    ```sh
    python main_file.py
    ```

