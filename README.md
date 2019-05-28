# WriterIdentification
Writer identification based on CASIA-HWDB1.1

## train the cnn

python set_divide_method1.py

set_divide_method1.py extracts pics from gnt files and resize them, finally divide the data set into 3 dirs(method 1).




python train.py

train.py train the cnn.

## visualize the cnn predict

With flask web page.

python hi_app.py

## visualize the cnn layer output

python visualization.py

