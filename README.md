# classification_resnet
Build a classification model with torchvision.models. The version of pytorch is 1.0

Firstly:
You should create a folder named data, then create three folders named train, val and test respectfully.

Secondly:
Go to the train(same in val,test) and creat the folders with your dataset, name the folders according to your class of the dataset.
For example, I have 2 categories(cat and dog), I would build two folders under the train(val,test) and name them cat, dog.

Thirdly:
Run the my_train.py.You will get a model named test.pth.

Finally:
If you want to predict a picture, you can run my_test.py.
