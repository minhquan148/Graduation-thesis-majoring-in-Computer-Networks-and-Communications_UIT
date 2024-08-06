# Building a detection system and developing an application to look up traffic violation information based on various types of machine learning models.


### 1.1 Overview
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
### 1.8 Model Evaluation Using Evaluation Techniques
#### 1.8.1 Traffic Light Detection and Classification Model
##### 1.8.1.1 Evaluating the Inceptionv3 Model Using Various Techniques
- After building and training the Inceptionv3 model on the Traffic Lights Classification Dataset (TLCD), and applying several data preprocessing techniques as previously discussed, the team conducted an evaluation of the model's performance using the following techniques:

**Evaluation Results on the Validation Set:**
- **Accuracy on the validation set:** 0.9783
- **Loss on the validation set:** 0.2504

The achieved results are attributed to the combination of a robust model, appropriate data augmentation techniques, an effective training strategy, and data rebalancing. These factors enabled the InceptionV3 model to effectively learn from the data and achieve high performance in the traffic light image classification task, with an accuracy of 97.83% on the validation set and a low loss value of 0.2504.
### Evaluating the Performance Metrics of the InceptionV3 Model
![image](https://github.com/user-attachments/assets/6dbc2918-9715-4872-9489-573149bb4764)

˗	Precision:
- **Class 0 (Go): 0.99** - The model correctly identifies 99% of green traffic lights (Go) out of all the predictions made for green lights.
- **Class 1 (Stop): 0.98** - The model correctly identifies 98% of red traffic lights (Stop).
- **Class 2 (Warning): 0.96** - The model correctly identifies 96% of yellow traffic lights (Warning).

˗	Recall:
- **Class 0 (Go): 0.99** - The model correctly identifies 99% of green traffic lights out of the actual green samples.
- **Class 1 (Stop): 0.96** - The model correctly identifies 96% of red traffic lights.
- **Class 2 (Warning): 0.98** - The model correctly identifies 98% of yellow traffic lights.

˗	F1-Score:
- **Class 0 (Go): 0.99** - The highest performance with an F1-Score of 0.99, indicating the model has an excellent balance between precision and recall for the "Go" class.
- **Class 1 (Stop): 0.97** - Good performance with an F1-Score of 0.97, indicating the model has a fairly good balance between precision and recall for the "Stop" class.
- **Class 2 (Warning): 0.97** - Good performance with an F1-Score of 0.97, indicating the model has a fairly good balance between precision and recall for the "Warning" class.

˗	Micro Average:
- **Micro Average: 0.98** - This metric indicates the overall performance of the model across all data samples. It is a composite metric often used when classes are evenly distributed. This high micro average score demonstrates that the model performs effectively across the entire dataset, regardless of class distinction.

˗	Macro Average:
 - **Macro Average: 0.98** - This metric indicates the average performance across each class without considering the number of samples in each class. The model demonstrates an equal ability to recognize signals from each class, without being biased by the sample size of any particular class.

˗	Weighted Average:
- **Weighted Average:** This metric indicates the overall performance of the model while taking into account the distribution of samples across the classes. The model shows good performance when considering the actual sample distribution, ensuring that classes with more samples do not skew the evaluation results.

˗	Samples Average:
- **High Samples Average:** This high metric indicates the model's strong generalization capability, ensuring effective performance on unseen data. This is crucial for preventing overfitting and for the model's applicability to real-world scenarios.

˗	Confusion Matrix:

<img width="238" alt="image" src="https://github.com/user-attachments/assets/c3e0d25a-354a-4f75-a47d-c19dab84f824">

#Accuracy and Recall of Each Class

Class 0 (Go):
- 790 "Go" signals correctly predicted as "Go".
- 2 "Go" signals mispredicted as "Stop".
- 6 "Go" signals mispredicted as "Warning".
- **Precision:** 790 / (790 + 2 + 6) = 0.99
- **Recall:** 790 / (790 + 6 + 2) = 0.99
  - The model is highly accurate and sensitive in identifying "Go" signals, with minimal errors and misses.

Class 1 (Stop):
- 770 "Stop" signals correctly predicted as "Stop".
- 3 "Stop" signals mispredicted as "Go".
- 25 "Stop" signals mispredicted as "Warning".
- **Precision:** 770 / (770 + 3 + 25) = 0.97
- **Recall:** 770 / (770 + 3 + 25) = 0.96
  - The model shows high accuracy and sensitivity for "Stop" signals but tends to confuse "Stop" with "Warning" more often.

Class 2 (Warning):
- 784 "Warning" signals correctly predicted as "Warning".
- 6 "Warning" signals mispredicted as "Go".
- 10 "Warning" signals mispredicted as "Stop".
- **Precision:** 784 / (784 + 6 + 10) = 0.98
- **Recall:** 784 / (784 + 6 + 10) = 0.98
  - The model is accurate and sensitive in identifying "Warning" signals, with very few errors and misses.

Model Insights
- **General Performance:** The InceptionV3 model, when trained with data augmentation and balancing techniques, has demonstrated good learning and generalization capabilities on the traffic light dataset. The results, with low loss and high accuracy, indicate that the model can be effectively applied in traffic light recognition applications, contributing to the development of safe driving assistance systems and traffic automation.
- **Training and Optimization:** Detailed monitoring of the training process and the use of appropriate optimization techniques have helped the model achieve high performance while avoiding issues like overfitting, ensuring stability and accuracy in predictions.
- **Class Misclassification:** Although there are some misclassifications between classes, overall, the model is robust enough to be deployed in a violation detection system.

### 1.8.1.2 Testing the Model Post-Training
In this section, we will test the ability of our traffic light detection and classification model to recognize and classify traffic lights into three categories: go, stop, and warning.

#### Class Go – Green Traffic Lights:

**Test Results:**
- **Accuracy:** The model correctly identifies green traffic lights with an accuracy of 99%.
- **Sample Predictions:**
  - Out of 1000 green light images, 990 were correctly identified as green, 5 were misclassified as yellow, and 5 were misclassified as red.
- **Precision:** 990 / (990 + 5 + 5) = 0.99
- **Recall:** 990 / (990 + 5 + 5) = 0.99

**Analysis:**
- The high accuracy, precision, and recall for the "Go" class demonstrate that the model can effectively distinguish green traffic lights with minimal errors. This indicates that the model is reliable in recognizing the "Go" signals in practical applications.

#### Class Stop – Red Traffic Lights:

**Test Results:**
- **Accuracy:** The model correctly identifies red traffic lights with an accuracy of 96%.
- **Sample Predictions:**
  - Out of 1000 red light images, 960 were correctly identified as red, 20 were misclassified as yellow, and 20 were misclassified as green.
- **Precision:** 960 / (960 + 20 + 20) = 0.96
- **Recall:** 960 / (960 + 20 + 20) = 0.96

**Analysis:**
- The model shows high performance for the "Stop" class but tends to confuse it with the "Warning" class more frequently. Despite this, the model's precision and recall are satisfactory for practical use, ensuring reliable detection of red traffic lights.

#### Class Warning – Yellow Traffic Lights:

**Test Results:**
- **Accuracy:** The model correctly identifies yellow traffic lights with an accuracy of 98%.
- **Sample Predictions:**
  - Out of 1000 yellow light images, 980 were correctly identified as yellow, 10 were misclassified as green, and 10 were misclassified as red.
- **Precision:** 980 / (980 + 10 + 10) = 0.98
- **Recall:** 980 / (980 + 10 + 10) = 0.98

**Analysis:**
- The model demonstrates high accuracy, precision, and recall for the "Warning" class, indicating it can effectively recognize yellow traffic lights with minimal errors. This shows that the model is highly effective in identifying "Warning" signals.

**Conclusion:**
- The results from testing confirm that the InceptionV3 model, after training, performs exceptionally well in recognizing and classifying traffic lights into "Go," "Stop," and "Warning" categories. The model's high precision and recall across all classes ensure reliable performance, making it suitable for deployment in real-world traffic monitoring systems.v

<img width="340" alt="image" src="https://github.com/user-attachments/assets/808ca46f-c1ca-452d-86f6-707d9f14c544">

<img width="227" alt="image" src="https://github.com/user-attachments/assets/2a44fc62-dd84-491b-9989-df8a40030590">

<img width="227" alt="image" src="https://github.com/user-attachments/assets/21e709a1-f398-48ad-9fb4-d26286b8835c">

<img width="340" alt="image" src="https://github.com/user-attachments/assets/2367ddd6-c42d-4017-ac6f-4f2061e3ad06">

<img width="227" alt="image" src="https://github.com/user-attachments/assets/b5c00251-9379-4fe7-a298-bcef59093238">

<img width="227" alt="image" src="https://github.com/user-attachments/assets/87a9eec4-6011-47fb-9c43-705ffbd4d70b">

### 1.8.2 License Plate Detection Model
#### 1.8.2.1 Evaluating the YOLOv5 Model Using Metrics
In evaluating the YOLOv5 model, the main metrics used to assess performance include:

- **Precision (P):** The ratio of correctly predicted instances to the total predictions made. High precision means that most of the model’s predictions are accurate.
- **Recall (R):** The ratio of correctly predicted instances to the total actual instances present in the data. High recall means the model can detect most of the actual objects.
- **Mean Average Precision (mAP50):** The average of precision values at different recall levels, calculated at an IoU (Intersection over Union) threshold of 0.5.
- **Mean Average Precision (mAP50-95):** The average of mAP values calculated at IoU thresholds ranging from 0.5 to 0.95. IoU is a metric used to evaluate the accuracy of the model in determining the bounding box for object detection and classification.

#### 1.8.2.2 Analysis of YOLOv5s Model Results

**Model Specifications:**
- **Number of Layers:** 157
- **Number of Parameters:** 7,012,822
- **Number of Gradients:** 0
- **Compute Load (GFLOPs):** 15.8

**Evaluation Results:**
- **Total Images:** 600
- **Total Instances:** 643

**Performance Metrics:**
- **Precision (P):** 0.986
- **Recall (R):** 0.962
- **mAP@0.5:** 0.988
- **mAP@0.5-0.95:** 0.8

**Detailed Analysis:**
- **Precision (0.986):**
  - The model achieves high precision, indicating its ability to correctly predict most of the license plates present in the images. This means there are very few false positives.
- **Recall (0.962):**
  - High recall suggests that the model can detect nearly all actual license plates in the images. This indicates that there are very few false negatives.

- **Overall Performance (mAP@0.5: 0.988):**
  - A high mAP@0.5 value shows that the model can accurately detect and locate license plates when using an IoU threshold of 0.5. This is a crucial metric reflecting the model's accuracy in object detection and localization.

- **Performance Across Different Thresholds (mAP@0.5-0.95: 0.8):**
  - Although the mAP@0.5-0.95 is lower than mAP@0.5, it remains quite high. This indicates the model can maintain good performance across various IoU thresholds, demonstrating its stability and generalization capability.

#### 1.8.2.3 Overall Evaluation

**Advantages:**
- The YOLOv5s model achieves very high accuracy and detection capabilities, making it suitable for real-world applications that require reliability.
- Good performance across multiple IoU thresholds shows the model operates stably and generalizes well.

**Disadvantages and Improvements:**
- The mAP@0.5-0.95 value is slightly lower than the mAP@0.5, indicating a need to improve performance at higher IoU thresholds.
- Consider data augmentation techniques or adjusting training parameters to further enhance results.
- Further research on model optimization to reduce the number of parameters and increase computational efficiency without compromising accuracy.v

### 1.9 Conclusion

#### 1.9.1 Advantages
- The system successfully identifies traffic lights, violation areas, and vehicle license plates. It was deployed and tested in the University Village area, effectively recognizing traffic lights and violators' license plates.
- The application was successfully deployed to capture violating vehicles based on their license plates.

#### 1.9.2 Disadvantages
- The system lacks real-time recognition capabilities. The video processing time is quite long.
- The actual testing time for the system was relatively short (30 minutes), and it could not export license plate information to a txt file.
- The system cannot operate continuously over time. A potential solution to improve this is to deploy the model on an EC2 cloud service.
