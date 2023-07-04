from evaluator import Evaluator
from trainer import Trainer


batch_size = 32
epochs = 10

from custom_datasets import Voc



trainer = Trainer(epochs, batch_size)
trainer.initialize_experiment()
trainer.fit("output/temp.h5")

'''
evaluator = Evaluator(batch_size)
evaluator.initialize()
evaluator.evaluate("output/temp.h5")
'''

from models.decoder import get_decoder_model
from utils import bbox_utils, data_utils, train_utils, eval_utils,drawing_utils
from models.ssd_mobilenet_v2 import get_model
from custom_datasets.voc import labels

test, test_size = Voc.get_custom_data_generator('./train')
imgs, gt_boxes, gt_labels = next(test)

hyper_params = train_utils.get_hyper_params()

_labels = ['bg'] + labels
hyper_params["total_labels"] = len(_labels)

# We calculate prior boxes for one time and use it for all operations because of the all images are the same sizes
prior_boxes    = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])

ssd_model = get_model(hyper_params)

ssd_model.load_weights("output/temp.h5")

ssd_decoder_model = get_decoder_model(ssd_model, prior_boxes, hyper_params)

print(f'img shape: {imgs[0].shape}')

pred_bboxes, pred_labels, pred_scores = ssd_decoder_model.predict(imgs,verbose=1)

print(pred_bboxes)
drawing_utils.draw_img_prediction(imgs[0],pred_bboxes[0],pred_labels[0],pred_scores[0],_labels,batch_size)

