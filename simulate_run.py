from evaluator import Evaluator
from trainer import Trainer

batch_size = 32
epochs = 1

from custom_datasets import Voc

trainer = Trainer(epochs, batch_size)
trainer.initialize_experiment()
#trainer.fit("output/temp.h5")

#evaluator = Evaluator(batch_size)
# evaluator.load_data(voc)
# evaluator.evaluate("output/voc_150epochs_64batches.h5")
