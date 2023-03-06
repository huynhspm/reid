import torchreid
from aic2023 import AIC2023
from torchreid.reid.models import build_model
from torchreid.reid.data import ImageDataManager
from torchreid.reid.engine import ImageTripletEngine, ImageSoftmaxEngine
from torchreid.reid.utils import load_pretrained_weights
from torchreid.reid.data.datasets import register_image_dataset
from torchreid.reid.optim import build_lr_scheduler, build_optimizer
import os
import mmpose


datamanager = ImageDataManager(root='./data',
                               sources='dukemtmcreid',
                               height=384,
                               width=384,
                               train_sampler='RandomIdentitySampler',
                               batch_size_train=32,
                               batch_size_test=32)

model = build_model(name='osnet_x1_0',
                    num_classes=10,
                    loss="softmax",
                    pretrained=False)

model = model.cuda()
dir_path = './logs/osnet_x1_0/dukemtmcreid'
checkpoint = 'osnet_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
# checkpoint = 'osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'

weight_path = os.path.join(dir_path, checkpoint)

load_pretrained_weights(model, weight_path)

optimizer = build_optimizer(model, optim="adam", lr=0.0003)
scheduler = build_lr_scheduler(optimizer,
                               lr_scheduler="single_step",
                               stepsize=20)
 
engine = ImageSoftmaxEngine(datamanager,
                            model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            use_gpu=True,
                            label_smooth=True)

# engine = ImageTripletEngine(datamanager,
#                             model,
#                             optimizer=optimizer,
#                             margin=0.3,
#                             weight_t=1,
#                             weight_x=1,
#                             scheduler=scheduler,
#                             use_gpu=True,
#                             label_smooth=True)

engine.run(save_dir=dir_path,
           max_epoch=10,
           start_epoch=0,
           print_freq=10,
           start_eval=0,
           eval_freq=10,
           test_only=True,
           dist_metric='euclidean',
           normalize_feature=False,
           visrank=True,
           visrank_topk=10,
           ranks=[1, 5, 10, 20],
           rerank=False)
