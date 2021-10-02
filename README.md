# Image classification and Localization

Given lung x-ray images, can you tell if a patient has TB or not

1. download the data by running the get_data shell script ```sh get_data.sh```

2. train the model. ```python train.py -c config.json```

3. start tensorboad ```tensorboard --logdir saved/log/``` and 
visit ```http://localhost:6006``` on your browser to see the training metrics
