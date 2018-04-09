# 日本語の解説記事はこちら(Japanese article is Here.)
[OpenFace+機械学習で視線検知](https://qiita.com/29Takuya/items/5c95b2a1fa42975305c9)

# 0. Background
This article is about what I've worked when I was undergraduate.  

Since the aim of this activity is to learn machine learning technology, there would be no novelty.

# 1. Goal
My goal was to detect eyesight from one face image.  

I mean whether the person in the image looks camera or not.

Some people in lab, where I worked, have done research on interactive robots system.

So I hoped my work would contribute to their interactive system, utilizing eyesight detection.

![data.png](https://qiita-image-store.s3.amazonaws.com/0/170782/a0c0a005-ca95-6340-1323-e7c44eb0b3b2.png)


# 2. Related Technology(OpenFace)
There are various images of face, for example with respect to size and position.

For that reason, I tried to detect eyesight after detection of face with [OpenFace API](https://cmusatyalab.github.io/openface/).

I used the API for following two objects.

1. Get position(coordinates) of faces's landmarks, black points in the image below.
2. Crop face in the shape of box.

![openface.jpg](https://qiita-image-store.s3.amazonaws.com/0/170782/e4aed818-6c38-4776-a612-fd0ce8a8d5a5.jpeg)

As you can see in the following image, 64 face's landmarks are pointed.

Then, cropping face area using points around eyes.

You can also get coordinates of each landmark.

<img src="https://qiita-image-store.s3.amazonaws.com/0/170782/92dfba6c-2ce7-55c5-464a-f32f5b6c5f90.png" width=60%>

# 3. Model for Prediction
I tested 3 following models to detect eyesight.

1. SVM(Support Vector Machine) with raw pixels as input.
2. SVM with SIFT features as input.
3. CNN(Convolutional Neural Network) with raw pixels as input.

I'm going to show each model below.

### 1. SVM(Support Vector Machine) with raw pixels as input.
As preprocessing, I applied histogram flattening to face images which cropped by OpenFace.

Then, I trained SVM with top one third of cropped images as input.

I show the example below.

![raw_svm.png](https://qiita-image-store.s3.amazonaws.com/0/170782/ccfa9268-afbe-cba8-ee09-b7968e3fbcb5.png)
The both Process of transfer to grayscale and use of top one third aim for dimension reduction. 

And histogram flattening is for robustness of divergence of contrast.

Finally, I trained SVM, using processed vectors as input.

### 2. SVM with SIFT features as input.
SIFT features is often used for image recognition.  

In my case, since I knew coordinates of face's landmarks around eyes, I calculated SIFT features in those points.

![sift.png](https://qiita-image-store.s3.amazonaws.com/0/170782/fd23f0cb-e732-24d1-315c-6027000b24fd.png)
I trained SVM using features obtained in that way.

### 3. CNN(Convolutional Neural Network) with raw pixels as input.

Input is top one third of processed face image, same as model 1(SVM with raw pixels).

I show network architecture below.

![cnn.png](https://qiita-image-store.s3.amazonaws.com/0/170782/8dd288cc-ab14-decc-6722-83fd08870b23.png)

# 4. Evaluation Experiments
### Dataset
The dataset I used for experiments includes 1300 positive images(looking camera) and 600 negative images(not looking camera).

I evaluate in hold-out way, divide whole images in the ration of  9 to 1.

![dataset-min.png](https://qiita-image-store.s3.amazonaws.com/0/170782/fd5e89d0-215b-7871-7fbf-55a75d311cba.png)


### Results
| Model                        | Accuracy [%] |
| :--------------------------- | -----------: |
| 1. SVM(Input: Raw pixel)     |         81.3 |
| 2. SVM(Input: SIFT Features) |         82.3 |
| 3. CNN                       |         88.7 |

### Conclusion（Impression）
Neural Network outperformed while the model was easy to deploy.

# 5. Appendix
I applied trained models to movie captured by web camera.

The blue box means the model predicts a person looks camera, and red means otherwise.

### 1. SVM(Input: Raw Pixel)
![output_svm.gif](https://qiita-image-store.s3.amazonaws.com/0/170782/de659fbf-fd5c-f930-e4c9-cb9467a4ae1f.gif)

### 2. CNN
![output_cnn.gif](https://qiita-image-store.s3.amazonaws.com/0/170782/a812ab3a-103c-0853-0397-0142445bf8cc.gif)

I had a feeling that CNN model is more stable than another.

As I thought, prediction seems to be difficult when the person doesn't face front.

Maybe, this is because we didn't have a sufficient number of images for training.

In addition, if we apply these models for movie, we should use techniques such as smoothing or sequential modeling.

I'm going to organize my code and upload to Github.

Thank you for reading!
