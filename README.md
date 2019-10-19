# classification_resnet
Build a classified model with torchvision.models. The version of pytorch is 1.0
                                                                                                                                            
***Firstly:***                                                                                                                                                                      
You should create a folder named data,     then create three folders named train, val and test respectfully.     
For example,                                                                       
	
	./                                                                                                                                           
	|                                                                                                                                       
	----data ————train----class 1 --- m1 * img                                                                                                                                     
	      |        |                                                                                                                                       
	      |        |------class 2 --- m2 * img                                                                                                                                     
	      |        |                                                                                                                                      
	      |        |------...                                                                                                                                      
	      |         ------class n --- m3 * img                                                                                                                                     
	      |                                                                                                                                      
	      |——————val------class 1 --- m4 * img                                                                                                                                      
	      |      |                                                                                                                                       
	      |      |--------class 2 --- m5 * img                                                                                                                                      
	      |      |                                                                                                                                      
	      |      |--------...                                                                                                                                      
	      |       --------class n --- m6 * img                                                                                                                                      
	      |                                                                                                                                      
 	      ——————test------class 1 --- m7 * img                                                                                                                                      
 	              |                                                                                                                                       
	              |-------class 2 --- m8 * img                                                                                                                                      
	              |                                                                                                                                      
	              |-------...                                                                                                                                      
	               -------class n --- m9 * img                                                                                                                                      
                                                                                                                                      

***Secondly:***                                                                                                                                                          
Go to the train(same in val,test) and creat the folders with your dataset, name the folders according to your class of the dataset. 
For example, I have 2 categories(cat and dog), I would build two folders under the train(val,test) and name them cat, dog.                    
                                                                                                                                          
***Thirdly:***                                                                                                                                         
	
	'''train1.py'''
	total_epochs = 1             #you can change it to any number 
	lr = 0.01
	delay_lr = 0.1
	delay_epoch_lr = 15          #every 15 epochs, the lr would be lr * delay_lr 
	save_dir = './data/test.pth' #save the model to your dir
	data_dir = './data'          #the data for train and test
	bs = 32 
You can change the parameters above. Then run the train1.py. You will get a model named test.pth.                    
                                                                                                                                        
***Finally:***                                                                                                                                     
If you want to predict a picture, you can run test1.py.               
