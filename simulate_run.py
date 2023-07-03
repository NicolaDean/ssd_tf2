from evaluator import Evaluator
from trainer import Trainer

batch_size = 32
epochs = 10

from custom_datasets import Voc

trainer = Trainer(epochs, batch_size)
trainer.initialize_experiment()
#trainer.fit("output/temp.h5")

evaluator = Evaluator(batch_size)
evaluator.initialize()
evaluator.evaluate("output/temp.h5")
