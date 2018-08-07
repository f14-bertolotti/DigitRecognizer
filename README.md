# DigitRecognizer  
A 0 to 9 digit recognizer written in python-keras with mnist dataset  

dependencies: kerar, numpy, opencv, tensorflow.  

running prebuilt models examples (recognizer.py):  

python3 recognizer.py -f ./mnist_models/MLPNN_type0-batch512-balanced-committee20/ -v    
python3 recognizer.py -f ./mnist_models/MLPNN_type0-batch512-balanced-committee20/    
python3 recognizer.py   

![Alt text](./images/1.png?raw=true "Example")  

training models examples (trainer.py):  

python3 trainer.py -s 512 -m "MLPNN_type0" -e 100 -c 20 -b  
python3 trainer.py -s 128 -m "CNN2D_type0" -e 20 -c 10 -b  
python3 trainer.py -s 512 -m "MLPNN_type0" -e 100 -c 20  
python3 trainer.py -s 128 -m "CNN2D_type0" -e 20 -c 10  
python3 trainer.py -s 512 -m "MLPNN_type0" -e 100  
python3 trainer.py -s 128 -m "CNN2D_type0" -e 20  
python3 trainer.py  

