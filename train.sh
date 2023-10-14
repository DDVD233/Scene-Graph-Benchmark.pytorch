source ~/miniconda3/bin/activate
conda activate svl_new

CUDA_VISIBLE_DEVICES=0,1 WORKING_DIRECTORY=/home/dvd/unbiased python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/cba/mask_rcnn_X_101_32x8d_FPN_3x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/ddavid/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/ddavid/cba/det/detectron2/output/model_0064999.pth OUTPUT_DIR /home/ddavid/unbiased/checkpoints/motif-precls-exmp