# drowsiness_detection_cnn
Project for our 3rd year minor defense created by our team of: me and Sabina Thapa ( https://github.com/SABINAKSHETTRY )

# Dataset
1) For eyes: dataset is custom
2) For yawn: dataset is mixture of images from kaggle and collected by us

# Processing dataset
1) image are gathered and the ROI is cropped from images and then splitted manually for test and train
2) Then data augmentation is performed to increase the number of images

# Model
CNN model as:
1) Convolution layer
    It consist of 3 2D convolution layer and used MaxPooling
2) Flatten layer:
    Output from convolution layer is fed to flatten layer
3) Fully connected layer:
    It consist of 3 layers. First one is input layer where the output from flatten layer is fed
    Second one is hidden layer and the last one output layer as sigmoid as activation function
    which gives result in between 0 and 1
    
# How To Run
1) Git clone repo using
    ```sh 
    https://github.com/undef125/drowsiness_detection_cnn.git
    ```
2) Change directory to our project:
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

