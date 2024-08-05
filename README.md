<img width="1317" alt="Screenshot 2024-08-06 at 01 04 28" src="https://github.com/user-attachments/assets/ff27326f-17f9-4465-b14a-ed1ca95078f4">### 1.1 Overview
- With the rapid increase in the number of vehicles participating in traffic, the infrastructure and management agencies responsible for traffic order are unable to keep up with this growth. Consequently, traffic violation detection has become a highly attractive field, aiming to improve the effectiveness of monitoring and handling traffic violations.
- Given this opportunity, Machine Learning (ML) and Deep Learning (DL) technologies have enhanced computers' ability to intelligently recognize, understand, and process images. Through training, ML models can identify objects and interpret images in an automated manner similar to how humans perceive and process them. In such systems, computer vision helps to identify and localize key image areas that need to be recognized, such as traffic lights, signals, stop lines, and vehicle license plates.
- The combination of Deep Learning and computer vision aims to detect traffic violations based on traffic signals. However, this research faces several challenges, including designing an appropriate Deep Learning model, the capability of object recognition, and optimizing performance and accuracy for practical applications.
- In summary, research into computer vision and Deep Learning models for building smart traffic systems capable of detecting traffic violations at traffic lights, especially at the Vietnam National University - Ho Chi Minh City (VNU-HCM), significantly contributes to traffic safety and minimizes potential accidents.

### 1.2 Reasons for Choosing the Topic
- Not yet widely implemented at VNU-HCM, allowing for deployment and application at this location.
- Effectively addresses current issues: non-compliance with traffic laws often leads to accidents.
- Saves human resources, with the system capable of automatically monitoring and detecting violations.

### 1.3 Research Objectives
The main purpose of this study is to develop a "SYSTEM FOR TRAFFIC VIOLATION DETECTION AND INFORMATION QUERY APPLICATION BASED ON MACHINE LEARNING MODELS" to automate the process of issuing fines and maintaining traffic order. The system will utilize various Deep Learning models to process data collected from surveillance cameras, allowing for automatic handling.

### 1.4 Research Subjects
- **Data Collection:**
  - Traffic Light Images: Initial data collection includes images of three types of traffic lights (go, stop, warning – green, red, yellow).
  - License Plate Images: Initial data collection includes images of various license plates from different locations.
  - Data Labeling: The labeling process identifies license plates in each image, creating a labeled dataset for model training.
- **Building and Developing Deep Learning Models:**
  - Using deep learning models such as SSD MobileNet v2, Inceptionv3 to detect and classify traffic lights and violating vehicles.
  - Using the Hough Line Transform algorithm to detect stop lines at red lights.
  - Using the YOLOv5 model to recognize license plates.
  - Using OCR modules to detect characters on license plates.

### 1.5 Research Scope
- **Offline Phase:**
  - Building the Dataset: Collecting, labeling, and augmenting data to create a rich and diverse dataset.
  - Training Models: Using the constructed dataset to train and evaluate the SSD MobileNet v2 and YOLOv5 models.
- **Online Phase:**
  - Evaluation and Improvement: Continuously evaluating the system's performance in real-world environments and making improvements to ensure accuracy and efficiency.

### 1.6 Related Research
This section introduces related research to our topic both domestically and internationally.

#### 1.6.1 Domestic:
- This study used deep learning models, specifically YOLOv4, to locate license plates in images. The results showed high accuracy with a mean Average Precision (mAP) of up to 91% and a processing speed of 31.2 FPS, demonstrating the model's effective application in real-time license plate recognition systems.
- Proposing the use of the YOLOv4 model to detect and recognize traffic signs and signals. They used Jetson TX2 hardware to optimize training time. The data included 32 classes with over 1500 images collected from Los Angeles. The system achieved a mAP of 91% and a speed of 31.2 FPS on the test dataset.

#### 1.6.2 International:
- Proposing a traffic light recognition system using deep learning and prior maps on the IARA autonomous vehicle platform. They used the YOLOv3 model to detect and classify traffic light states from camera images. Results showed the system could accurately recognize relevant traffic lights on predefined routes in Vitoria city.
- This study used the Inception-V3 deep learning model based on transfer learning to detect and recognize traffic lights. The process included training and testing the model on the LISA traffic light dataset, augmented by data preprocessing methods. Results showed the model achieved 98.6% accuracy in recognizing traffic lights.

#### 1.6.3 Existing Issues
- Despite numerous studies aimed at improving the quality and efficiency of traffic violation monitoring systems, some issues remain unresolved:
  - **Model Accuracy and Performance:** While deep learning models like CNN and YOLO have achieved significant success, there are still issues regarding accuracy and performance in real-world applications, especially with noisy images and varying lighting conditions.
  - **Scalability and Flexibility:** Most studies focus on a specific model for a particular task, lacking integration to create a comprehensive system.

#### 1.6.4 Issues to Focus On
- Based on the identified existing issues, this research will focus on addressing the following:
  - Building a diverse and rich dataset: Collecting and labeling data for traffic light classification and license plate recognition.
  - Developing and optimizing models: Researching and training the Inceptionv3 model (for traffic light classification) and the YOLOv5 model (for license plate recognition).

 ### 1.7 Model Overview
- The study proposes a framework for a "Traffic Violation Vehicle Detection System," designed to ensure efficient and consistent data collection, processing, and analysis.

<img width="1290" alt="Screenshot 2024-08-06 at 01 04 28" src="https://github.com/user-attachments/assets/360e9689-2ad8-4b30-b1ec-8283d4536450">


- The system consists of three main components:
  - **Traffic Light:** This part focuses on building and developing datasets and machine learning models. The process includes data collection and labeling, data augmentation, and developing and training the InceptionV3 model to detect and classify traffic lights.
  - **License Plate Detection:** This part focuses on building and developing datasets and machine learning models. The process includes data collection and labeling, data augmentation, and developing and training models like YOLO to detect the position of license plates within a frame.
  - **Optical Character Recognition:** This part takes the results of the previous process to recognize characters and provide accurate information about the violating vehicle's license plate.
- Additionally, the "Traffic Violation Vehicle Detection System" also uses Hough Line Transform combined with SSD MobileNetv2 (for detecting vehicles entering violation zones) to analyze the behavior of monitored traffic participants.
- The workflow of the red light traffic violation detection system:
  - Data is collected from surveillance cameras placed at traffic lights.
  - The collected data is fed into the SSD MobileNetv2 model to detect traffic lights - the position of traffic lights within a frame.
  - After detecting the traffic lights, the Inceptionv3 model classifies the lights into three classes (green, yellow, red - go, warning, stop) and follows two scenarios:
    - **Scenario 1:** The Inceptionv3 model detects a green light, and no further actions are taken by the system.
    - **Scenario 2:** The Inceptionv3 model detects a red or yellow light, and the system continues with the violation detection processes.
  - After detecting the traffic light state as red or yellow (stop or warning):
    - Detect the stop line and display it on the frame using Probabilistic Hough Line Transform (HoughLinesP) (an improved version of Hough Line Transform).
    - Create a violation detection zone using SSD MobileNetv2 to identify vehicles crossing the stop line and entering the violation zone. This zone is developed from the stop line (meaning it detects the line and automatically determines the zone based on its parameters). When a vehicle enters the zone, it automatically detects and outputs an image of the vehicle.
    - After detecting and outputting the vehicle image, the YOLOv5 model detects the position of the vehicle's license plate.
- The OCR module recognizes and reads the characters on the violating vehicle's license plate, outputting a txt file containing the license plate information. The results are displayed visually for easy monitoring.
