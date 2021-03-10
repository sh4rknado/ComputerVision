from Model.extractEmbeddings import extractEmbeddings as em
from Model.TrainModel import TrainModel as tm
from  Model.recognize import Recognize

# Create Model
embedded = em()
embedded.run()

# Trainning Model
train_model = tm()
train_model.run()

# Run the recognize
recognize = Recognize()
recognize.run()
