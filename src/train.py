import torchreid
from aic2023 import AIC2023
from torchreid.reid.models import build_model
from torchreid.reid.data import ImageDataManager
from torchreid.reid.engine import ImageTripletEngine
from torchreid.reid.utils import load_pretrained_weights
from torchreid.reid.data.datasets import register_image_dataset
from torchreid.reid.optim import build_lr_scheduler, build_optimizer
register_image_dataset('aic2023', AIC2023)

datamanager = torchreid.data.ImageDataManager(root='./data',
                                              sources='aic2023',
                                              height=256,
                                              width=192,
                                              train_sampler='RandomIdentitySampler',
                                              batch_size_train=32,
                                              batch_size_test=32)

model = build_model(name='resnet50',
                    num_classes=datamanager.num_train_pids,
                    loss="triplet",
                    pretrained=True)
model = model.cuda()

optimizer = build_optimizer(model, optim="adam", lr=0.0003)
scheduler = build_lr_scheduler(optimizer,
                               lr_scheduler="single_step",
                               stepsize=20)

engine = ImageTripletEngine(datamanager,
                            model,
                            optimizer=optimizer,
                            margin=0.3,
                            weight_t=1,
                            weight_x=1,
                            scheduler=scheduler,
                            use_gpu=True,
                            label_smooth=True)
                            
engine.run(save_dir="logs/resnet50",
           max_epoch=10,
           start_epoch=0,
           print_freq=10,
           start_eval=0,
           eval_freq=10,
           test_only=False,
           dist_metric='euclidean',
           normalize_feature=False,
           visrank=False,
           visrank_topk=10,
           ranks=[1, 5, 10, 20],
           rerank=False)