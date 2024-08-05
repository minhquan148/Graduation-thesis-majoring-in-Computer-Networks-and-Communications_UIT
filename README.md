1.1	Tổng quan
-	Với số lượng các phương tiện tham gia giao thông cũng đang ngày một gia tăng một cách chóng mặt, cùng với đó các cơ sở hạ tầng và ban ngành quản lý về trật tự giao thông cũng đang không thể đáp ứng đủ với sự gia tăng đó. Chính điều này đã giúp cho nhận diện vi phạm giao thông là một lĩnh vực đang thu hút nhiều sự quan tâm, với mục tiêu cải thiện hiệu quả giám sát và xử lý các hành vi vi phạm giao thông.
-	Với cơ hội rộng mở như vậy, Công nghệ học máy (Machine Learning) và học sâu (Deep Learning) đã tăng khả năng cho máy tính nhận diện, hiểu và xử lý hình ảnh một cách thông minh. Nhờ việc học thông qua huấn luyện mà các mô hình máy học có thể nhận diện vật thể và hiểu hình ảnh theo cách tự động tương tự như cách con người nhìn và xử lý chúng. Trong hệ thống, thị giác máy tính giúp nhận diện và khoanh vùng được phạm vi hình ảnh cần nhận diện như đèn giao thông, tín hiệu đèn giao thông, vạch dừng và biển số xe.
-	Sự kết hợp giữa Học sâu và thị giác máy tính để nhận diện vi phạm giao thông dựa trên tín hiệu đèn giao thông. Tuy nhiên, nghiên cứu này đặt ra nhiều thách thức, từ việc thiết kế mô hình Học sâu phù hợp, khả năng nhận diện các vật thể, đến việc tối ưu hóa hiệu suất và độ chính xác khi ứng dụng thực tế.
-	Tóm lại, nghiên cứu về thị giác máy tính và mô hình Học sâu để xây dựng hệ thống giao thông thông minh, khả năng nhận diện được hành vi vi phạm giao thông tại các đèn giao thông đặc biệt tại vị trí Đại học Quốc Gia TP HCM, đóng góp phần nhiều vào lĩnh vực an toàn giao thông, hạn chế tối đa các tai nạn có thể xảy ra.
1.2	Lý do chọn đề tài
-	Chưa được triển khai rộng rãi tại Đại học Quốc Gia – TP HCM, có thể triển khai và ứng dụng tại địa điểm này.
-	Giải quyết được tối đa vấn đề thực tế đang xảy ra: không tuân thủ luật giao thông dễ gây nên tai nạn giao thông.
-	Tiết kiệm được sức người, hệ thống có thể tự động giám sát và phát hiện hành vi vi phạm.
1.3	Mục tiêu nghiên cứu
Mục đích chính của đề tài này là phát triển “XÂY DỰNG HỆ THỐNG NHẬN DIỆN VI PHẠM VÀ ỨNG DỤNG TRA CỨU THÔNG TIN VI PHẠM GIAO THÔNG DỰA TRÊN CÁC LOẠI MÔ HÌNH MÁY HỌC”, nhằm tự động hóa quá trình phạt nguội và giữ trật tự an toàn giao thông. Hệ thống sẽ sử dụng đa dạng các loại mô hình Deep Learning để xử lý dữ liệu được thu thập các camera giám sát, cho phép xử lý một cách tự động.
1.4	Đối tượng nghiên cứu
-	Thu thập dữ liệu: 
•	Hình ảnh đèn giao thông: Dữ liệu ban đầu được thu thập hình ảnh của 3 loại đèn giao thông (go, stop, warning – xanh, đỏ, vàng).
•	Hình ảnh biển số xe: Dữ liệu ban đầu được thu thập hình ảnh của biển số xe các loại ở nhiều nơi.
•	Gắn nhãn dữ liệu: Quá trình gán nhãn được thực hiện để xác định biển số xe trong từng hình ảnh, tạo ra một bộ dữ liệu đã được gán nhãn phục vụ cho việc huấn luyện mô hình.
-	Xây dựng và phát triển các loại mô hình Deep Learning:
•	Sử dụng các mô hình học sâu: SSD MobileNet v2, Inceptionv3 để nhận diện và phân loại đèn giao thông và phương tiện vi phạm.
•	Sử dụng thuật toán Hough Line Transform: nhận diện vạch dừng chờ đèn đỏ.
•	Sử dụng mô hình YOLOv5: nhận diện biển số xe.
•	Sử dụng module OCR để nhận diện ký tự trên biển số xe.
1.5	Phạm vi nghiên cứu
-	Phần offline:
•	Xây dựng bộ dữ liệu: Thu thập, gán nhãn và tăng cường dữ liệu để tạo ra một bộ dữ liệu phong phú và đa dạng.
•	Huấn luyện mô hình: Sử dụng bộ dữ liệu đã xây dựng để huấn luyện và đánh giá các mô hình SSD MobileNet v2, YOLOv5
-	Phần online: Đánh giá và cải tiến: Liên tục đánh giá hiệu suất của hệ thống trong môi trường thực tế và cải tiến để đảm bảo độ chính xác và hiệu quả.
1.6	Các nghiên cứu liên quan
˗	Phần này chúng tôi sẽ giới thiệu về các nghiên cứu liên quan đến đề tài của chúng tôi ở trong và ngoài nước.
1.6.1	Trong nước:
˗	Trong nghiên cứu này, nhóm đã sử dụng các mô hình học sâu, cụ thể là YOLOv4, để xác định vị trí biển số xe trong ảnh. Kết quả cho thấy mô hình đạt độ chính xác cao, với mAP (mean Average Precision) lên tới 91% và tốc độ xử lý đạt 31,2 FPS, cho thấy khả năng áp dụng hiệu quả của mô hình trong các hệ thống nhận diện biển số xe thời gian thực. 
˗	Đề xuất sử dụng mô hình YOLOv4 để phát hiện và nhận diện biển báo và tín hiệu đèn giao thông. Họ sử dụng phần cứng Jetson TX2 để tối ưu thời gian huấn luyện. Dữ liệu sử dụng bao gồm 32 lớp với hơn 1500 ảnh được thu thập từ Los Angeles. Hệ thống đạt được chỉ số mAP 91% và tốc độ 31.2 FPS trên tập dữ liệu kiểm tra. 
1.6.2	Ngoài nước
˗	Đề xuất một hệ thống nhận diện đèn giao thông sử dụng kỹ thuật học sâu (deep learning) và bản đồ trước (prior maps) trên nền tảng xe tự hành IARA. Họ sử dụng mô hình YOLOv3 để phát hiện và phân loại trạng thái đèn giao thông từ hình ảnh camera. Kết quả cho thấy hệ thống có thể nhận diện chính xác các đèn giao thông liên quan trong các tuyến đường được định trước tại thành phố Vitoria. 
˗	Trong nghiên cứu này, nhóm tác giả đã sử dụng mô hình học sâu Inception-V3 dựa trên phương pháp học chuyển giao để phát hiện và nhận diện đèn giao thông. Quá trình thực hiện bao gồm việc huấn luyện và kiểm tra mô hình trên bộ dữ liệu LISA traffic light, được tăng cường bởi các phương pháp tiền xử lý dữ liệu. Kết quả cho thấy mô hình đạt được độ chính xác 98,6% trong việc nhận diện đèn giao thông. 
1.6.3	Những vấn đề còn tồn tại.
-	Mặc dù có nhiều nghiên cứu đã được thực hiện nhằm cải thiện chất lượng và hiệu quả của các hệ thống giám sát vi phạm giao thông, vẫn còn tồn tại một số vấn đề chưa được giải quyết hoàn toàn:
•	Độ chính xác và hiệu suất của mô hình: Mặc dù các mô hình học sâu như CNN và YOLO đã đạt được nhiều thành công, vẫn còn tồn tại vấn đề về độ chính xác và hiệu suất khi áp dụng trong môi trường thực tế, đặc biệt là với các hình ảnh có nhiễu và điều kiện ánh sáng khác nhau.
•	Khả năng mở rộng và tính linh hoạt: Hầu hết các nghiên cứu tập trung vào một loại mô hình xử lý cho một công việc cụ thể nhất đinhhj, chưa có nhiều sự kết hợp để tạo nên một hệ thống.
1.6.4	Những vấn đề cần tập trung giải quyết.
-	Dựa trên các vấn đề còn tồn tại đã được xác định, nghiên cứu này sẽ tập trung vào việc giải quyết các vấn đề sau:
•	Xây dựng bộ dữ liệu đa dạng và phong phú: Thu thập và gán nhãn dữ liệu  cho 2 bộ dữ liệu phân loại đèn giao thông và nhận diện biển số  phương tiện vi phạm.
•	Phát triển và tối ưu hoá mô hình: Nghiên cứu, huấn luyện 2 mô hình  Inceptionv3 (cho nhiệm vụ phân loại đèn giao thông) và YOLOv5 (cho nhiệm vụ nhận diện biển số xe).
![Uploading image.png…]()
