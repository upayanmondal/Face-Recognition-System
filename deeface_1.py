import numpy as np
import os
import cv2
from deepface import DeepFace


cwd = os.getcwd()
Path = os.path.join(cwd,"dataset")

def detect_face(image_path):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    img = image_path
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_data = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    for x,y,w,h in face_data:
        face = img[y:y+h, x:x+w]
        l = [x,y,w,h]
        return (face,l)

def train_faces(image_path):

    encoding = {}
    
    for person in os.listdir(image_path):
        print(person)
        person_path = os.path.join(image_path, person)
        print(person_path)
        
        if os.path.isdir(person_path):
            print(os.listdir(person_path))
            
            i=1
            for img_name in os.listdir(person_path):
                print(img_name)
                img_path = os.path.join(person_path, img_name)
                print(img_path)

                try:

                    #face = detect_face(img_path)
                    embedding = DeepFace.represent(img_path, model_name= "Facenet")
                    print("face detected")
                    #encoding[img_path] = (person, embedding)
                    encoding[(person,i)] = (embedding[0]['embedding'])
                    #print(encoding)
                    i+=1

                except:
                    continue

    return encoding

def face_recognition(test_img, encoding):

    test_img = detect_face(test_img)

    try:
        test_embedding = DeepFace.represent(test_img[0], model_name= "Facenet")
        print("for test image    ",test_embedding,len(test_embedding))
        min_dst = float("inf")
        identity = "Unknown"

        for (person,i) , embedding in encoding.items():
            print(person,i)
            dist = np.linalg.norm(np.array(test_embedding[0]['embedding']) - np.array(embedding))
            print(123)

            if dist < min_dst :
                min_dst = dist
                if dist < 10 :
                    identity = person
                else:
                    identity = "Unknown"
        
        return identity

    except:
        return "No Face Detected"

    

if __name__ == "__main__":

    print("Har Har MAHADEV")
    print("...Train Faces...")
    trained_encoding = train_faces(Path)

    cam = cv2.VideoCapture(0)

    while True:

        ret, frame = cam.read()

        if ret == False:
            break
        
        data = detect_face(frame)

        if data != None:
            #coordinates = data[1]
            coordinates = data[1]

            person_name = face_recognition(frame, trained_encoding)

            x,y,w,h = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            cv2.putText(frame, person_name, (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            cv2.imshow("pic", frame)
            if(cv2.waitKey(1) & 0xFF) == ord('u'):
                break

        else:
            person_name = face_recognition(frame, trained_encoding)

            cv2.putText(frame, person_name, (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            cv2.imshow("pic", frame)
            if(cv2.waitKey(1) & 0xFF) == ord('u'):
                break

    cam.release()
    cv2.destroyAllWindows()
    
    #print(trained_encoding)    