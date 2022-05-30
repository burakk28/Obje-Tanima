
# -*-coding:utf-8-*-

#opencv kütüphanemizi dahil ediyoruz 
import cv2
import numpy as np

#resmi çağırıyoruz
img = cv2.imread ('test.jpg')


#resmin enini ve boyunu tuttuğumuz alan
img_width = img.shape [1]
img_height = img.shape [0]


img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB = True)
labels = ['insan', 'bisiklet', 'araba', 'motorsiklet', 'ucak', 'otobüs', 'tren', 'kamyon', 'tekne',
          'trafik lambasi', 'yangin muslugu', 'dur isareti', 'parkmetre', 'banka', 'kus', 'kedi',
          'kopek', 'at', 'koyun', 'inek', 'fil', 'ayı', 'zebra', 'zürafa', 'sırt çantası',
          'semsiye', 'el cantasi', 'kravat', 'bavul', 'frizbi', 'kayak', 'snowboard', 'spor topu',
          'ucurtma', 'beyzbol sopasi', 'beyzbol eldiveni', 'kaykay', 'sörf tahtasi', 'tenis raketi',
          'sise', 'bardak', 'kupa', 'catal', 'bicak', 'kasik', 'kase', 'muz', 'elma',
          'sandvic', 'portakal', 'brokoli', 'havuc', 'sosisli sandvic', 'pizza', 'cörek', 'kek', 'sandalye',
          'kanepe', 'saksi', 'yatak', 'yemek masasi', 'tuvalet', 'tvmonitor', 'laptop', 'mouse',
          'uzaktan', 'keyboard', 'cep telefonu', 'mikrodalga', 'firin', 'tost makinesi', 'lavabo', 'buzdolabi',
          'kitap', 'saat', 'vazo', 'makas', 'oyuncak ayi', 'fön makinesi', 'diş fircasi']


colors = ['0,0,0','0,255,0','255,0,255','0,255,255','0,0,255']   
colors = [np.array(color.split(',')).astype('int') for color in colors]
colors = np.array(colors)
colors = np.tile(colors,(20,1))

#hazır modellerimizi çağırıyoruz
#layers ile bütün katmanları çekiyoruz
model = cv2.dnn.readNetFromDarknet('model/yolov3.cfg','model/yolov3.weights')
layers = model.getLayerNames()
output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers() ]

model.setInput(img_blob)
#çıktı katmanlarımızı buraya koyuyoruz
detection_layers = model.forward(output_layer)

ids_list = []
boxes_list = []
confidences_list = []

#çıkan matrislerin içini tek tek geziyoruz  
for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores = object_detection [5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]
        #güven alanı %30 dan fazla ise objeyi çiz
        if confidence > 0.30 :
            
            label = labels[predicted_id]
            bounding_box = object_detection [0:4] * np.array([img_width,img_height,img_width,img_height])
            (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype('int')
             
            start_x = int (box_center_x - (box_width/2))
            start_y = int (box_center_y - (box_height/2))
            
            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x,start_y,int(box_width),int(box_height)])
            
#bütün yüksek güvenirliliğe sahip dikdörtgenleri bir liste biçiminde bize döndürüyor            
max_ids = cv2.dnn.NMSBoxes(boxes_list,confidences_list,0.5,0.4)
            
for max_id in max_ids:
    max_class_id = max_id
    box = boxes_list [max_class_id]
                
    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height = box[3]
                
    predicted_id = ids_list[max_class_id]
    label =labels[predicted_id]
    confidence = confidences_list[max_class_id]
                
             
             
    end_x = start_x + box_width
    end_y = start_y + box_height
             
    box_color = colors[predicted_id]
    box_color = [int(each)for each in box_color]
    
    #çerçevelerimizi çiziyoruz 
    cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,2)
    cv2.putText(img,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
            
               
        
cv2.imshow('Tespit Ekrani', img)
cv2.waitKey(0) 
cv2.destroyAllWindows() 
cv2.waitKey(1)

