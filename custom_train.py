from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

import augmentation
from models.ssd_mobilenet_v2 import get_model, init_model
from ssd_loss import CustomLoss
from utils import bbox_utils, data_utils, train_utils
from custom_datasets.voc import Voc,labels

import tqdm

#BACKBONE = "vgg16"
BACKBONE = "mobilenet_v2"

class Trainer:
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.fitter = None
        
    def initialize_experiment(self):
        hyper_params = train_utils.get_hyper_params(BACKBONE)
        _labels = ["bg"] + labels
        hyper_params["total_labels"] = len(_labels)

        train_data, train_size = Voc.get_custom_data_generator('./train')#augmentation_func=augmentation.apply
        val_data  , val_size   = Voc.get_custom_data_generator('./valid')

        # We calculate prior boxes for one time and use it for all operations because of the all images are the same sizes
        prior_boxes    = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
        ssd_train_feed = train_utils.custom_generator(train_data, prior_boxes, hyper_params)
        ssd_val_feed   = train_utils.custom_generator(val_data, prior_boxes, hyper_params)

        #print(next(ssd_train_feed))

        step_size_train = train_utils.get_step_size(train_size, self.batch_size)
        step_size_val   = train_utils.get_step_size(val_size, self.batch_size)


        ssd_model = get_model(hyper_params)
        ssd_custom_losses = CustomLoss(hyper_params["neg_pos_ratio"], hyper_params["loc_loss_alpha"])
        ssd_model.compile(optimizer=Adam(learning_rate=1e-3),
                          loss=[ssd_custom_losses.loc_loss_fn, ssd_custom_losses.conf_loss_fn])
        init_model(ssd_model)

        def fitter(output_path):
            checkpoint_callback = ModelCheckpoint(output_path, monitor="val_loss",
                                                  save_best_only=True,
                                                  save_weights_only=True)
            tensorboard_callback = TensorBoard()
            learning_rate_callback = LearningRateScheduler(train_utils.scheduler, verbose=0)
            ssd_model.fit(ssd_train_feed,
                          steps_per_epoch=step_size_train,
                          validation_data=ssd_val_feed,
                          validation_steps=step_size_val,
                          epochs=self.epochs,
                          callbacks=[checkpoint_callback, tensorboard_callback, learning_rate_callback])

        self.fitter = fitter

    def fit(self, output_path):
        self.fitter(output_path)