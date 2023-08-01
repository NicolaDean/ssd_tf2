from custom_train import Trainer

batch_size = 32
epochs = 10

from custom_datasets.voc import Voc

trainer = Trainer(epochs, batch_size)
trainer.initialize_experiment()
trainer.fit("output/temp.h5")