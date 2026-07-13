tree/
│
├── 00_fundamentals/
│   ├── convolution/
│   │   ├── conv2d.py
│   │   ├── depthwise_conv.py
│   │   └── pointwise_conv.py
│   │
│   ├── normalization/
│   │   ├── batch_norm.py
│   │   ├── layer_norm.py
│   │   └── group_norm.py
│   │
│   ├── activations/
│   │   ├── relu.py
│   │   ├── gelu.py
│   │   ├── silu.py
│   │   └── sigmoid.py
│   │
│   ├── pooling/
│   │   ├── max_pool.py
│   │   ├── avg_pool.py
│   │   └── adaptive_pool.py
│   │
│   ├── blocks/
│   │   ├── conv_bn_activation.py
│   │   ├── residual_block.py
│   │   ├── bottleneck_block.py
│   │   ├── squeeze_excitation.py
│   │   ├── attention_block.py
│   │   └── transformer_block.py
│   │
│   └── upsampling/
│       ├── interpolation.py
│       ├── conv_transpose.py
│       └── pixel_shuffle.py
│
├── 01_image_classification/
│   ├── convolutional_networks/
│   │   ├── lenet/
│   │   ├── alexnet/
│   │   ├── vgg/
│   │   ├── resnet/
│   │   ├── efficientnet/
│   │   ├── regnet/
│   │   └── convnext/
│   │
│   ├── vision_transformers/
│   │   ├── vit/
│   │   ├── swin_transformer/
│   │   ├── cvt/
│   │   ├── coatnet/
│   │   └── maxvit/
│   │
│   └── experiments/
│       ├── cnn_vs_vit.ipynb
│       ├── efficientnet_vs_convnext.ipynb
│       └── pretrained_finetuning.ipynb
│
├── 02_object_detection/
│   ├── two_stage_detectors/
│   │   ├── rcnn/
│   │   ├── fast_rcnn/
│   │   └── faster_rcnn/
│   │
│   ├── one_stage_detectors/
│   │   └── yolo/
│   │
│   ├── transformer_detectors/
│   │   └── detr/
│   │
│   └── experiments/
│       ├── yolo_vs_faster_rcnn.ipynb
│       ├── one_stage_vs_two_stage.ipynb
│       └── cnn_detector_vs_detr.ipynb
│
├── 03_instance_segmentation/
│   ├── mask_rcnn/
│   └── experiments/
│       └── faster_rcnn_vs_mask_rcnn.ipynb
│
├── 04_semantic_segmentation/
│   ├── encoder_decoder/
│   │   ├── unet/
│   │   └── unet_plus_plus/
│   │
│   ├── atrous_convolution/
│   │   └── deeplabv3_plus/
│   │
│   └── experiments/
│       ├── unet_vs_unetpp.ipynb
│       └── unet_vs_deeplabv3plus.ipynb
│
├── 05_sequence_modeling/
│   ├── recurrent_networks/
│   │   ├── rnn/
│   │   ├── lstm/
│   │   └── gru/
│   │
│   ├── sequence_to_sequence/
│   │   └── seq2seq_attention/
│   │
│   ├── transformers/
│   │   └── transformer/
│   │
│   └── experiments/
│       ├── rnn_vs_lstm.ipynb
│       ├── lstm_vs_gru.ipynb
│       └── seq2seq_vs_transformer.ipynb
│
├── 06_representation_learning/
│   ├── autoencoders/
│   │   ├── autoencoder/
│   │   └── variational_autoencoder/
│   │
│   ├── contrastive_learning/
│   │   ├── simclr/
│   │   └── clip/
│   │
│   └── experiments/
│       ├── ae_vs_vae.ipynb
│       └── supervised_vs_pretrained_features.ipynb
│
├── 07_generative_modeling/
│   ├── gan/
│   │   ├── vanilla_gan/
│   │   ├── dcgan/
│   │   ├── conditional_gan/
│   │   └── stylegan/
│   │
│   ├── diffusion/
│   │   ├── diffusion_probabilistic_models/
│   │   ├── ddpm/
│   │   └── ddim/
│   │
│   └── experiments/
│       ├── vae_vs_gan.ipynb
│       └── gan_vs_diffusion.ipynb
│
├── 08_text_to_image/
│   ├── stable_diffusion/
│   ├── latent_diffusion/
│   ├── text_encoder/
│   ├── vae/
│   ├── unet_denoiser/
│   └── scheduler/
│
├── 09_multitask_models/
│   ├── classification_detection_segmentation/
│   │   ├── shared_backbone/
│   │   ├── task_heads/
│   │   └── multitask_training/
│   │
│   └── experiments/
│       └── shared_backbone_multitask.ipynb
│
├── 10_pretrained_models/
│   ├── torchvision/
│   ├── timm/
│   ├── transformers/
│   └── finetuning/
│       ├── freeze_backbone.py
│       ├── partial_unfreeze.py
│       ├── full_finetune.py
│       ├── differential_learning_rate.py
│       └── warmup_scheduler.py
│
├── datasets/
│   ├── classification_dataset.py
│   ├── detection_dataset.py
│   ├── segmentation_dataset.py
│   ├── sequence_dataset.py
│   └── dataloader_utils.py
│
├── training/
│   ├── classification_trainer.py
│   ├── detection_trainer.py
│   ├── segmentation_trainer.py
│   ├── sequence_trainer.py
│   └── generative_trainer.py
│
├── evaluation/
│   ├── classification_metrics.py
│   ├── detection_metrics.py
│   ├── segmentation_metrics.py
│   ├── generation_metrics.py
│   └── visualization.py
│
└── README.md