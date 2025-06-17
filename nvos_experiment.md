# NVOS Experiment 

update: 2025-03-02

## horns_center

iou 0.9706069555476348
acc 0.9948868870464853

python .\gsd_nvos_render_views.py --scene=horns_center
python .\gsd_nvos_use_sam2.py --scene=horns_center
python gsd_nvos_obj_removal.py --scene=horns_center --out --with_var
--> data/mid/gsd/horns_center_crt_0.5_stride_1_radius_0_bb_0.0_var_0.2
--> data/mid/gsd/horns_center_crt_0.5_stride_1_radius_5_bb_0.0_var_0.2
python gsd_nvos_metrics.py --scene=horns_center --with_var 

iou 0.9669856401810455
acc 0.9942786458333334

## horns_left 

iou 0.9256143011888931
acc 0.9953985838162426

python .\gsd_nvos_render_views.py --scene=horns_left
python .\gsd_nvos_use_sam2.py --scene=horns_left
python gsd_nvos_obj_removal.py --scene=horns_left --out --with_var
--> data/mid/gsd/horns_left_crt_0.5_stride_1_radius_0_bb_0.0_var_0.2
python gsd_nvos_metrics.py --scene=horns_left --with_var 

iou 0.9410305869385506
acc 0.996321875

last 

iou 0.9412250895399027
acc 0.9963333333333333

## fern 

iou 0.8203410279612124
acc 0.9416770833333333

iou 0.8237872509307209
acc 0.9428807291666667

python .\gsd_nvos_render_views.py --scene=fern
--> data/gsd/fern/frames/1.jpg --- 19.jpg
python .\gsd_nvos_use_sam2.py --scene=fern 2s
--> data/gsd/fern/frames/masks/..png
python gsd_nvos_obj_removal.py --scene=fern --out --with_var
--> data/mid/gsd/fern_crt_0.5_stride_1_radius_0_bb_0.0_var_0.2
python gsd_nvos_metrics.py --scene=fern --with_var 

iou 0.8220298794434462
acc 0.942534375

## flower 

iou 0.9554418402740591
acc 0.9897632760665995

iou 0.9575961089952314
acc 0.9902463541666666

iou 0.9580670808245797
acc 0.9904130208333334

python .\gsd_nvos_render_views.py --scene=flower
python .\gsd_nvos_use_sam2.py --scene=flower
python gsd_nvos_obj_removal.py --scene=flower --out --with_var
--> data/mid/gsd/flower_crt_0.5_stride_1_radius_0_bb_0.0_var_0.2
python gsd_nvos_metrics.py --scene=flower --with_var 

iou 0.9579802243789917
acc 0.9903984375

## fortress

iou 0.9538933454019936
acc 0.9913520867451919

iou 0.9560047199278129
acc 0.991746875

python .\gsd_nvos_render_views.py --scene=fortress
python .\gsd_nvos_use_sam2.py --scene=fortress
python gsd_nvos_obj_removal.py --scene=fortress --out --with_var
--> data/mid/gsd/fortress_crt_0.5_stride_1_radius_0_bb_0.0_var_0.2
python gsd_nvos_metrics.py --scene=fortress --with_var 

iou 0.9711842929750052
acc 0.9946109375

## leaves
 
iou 0.93062803729991
acc 0.9956575898106156

iou 0.9346237310119478
acc 0.9959046875

python .\gsd_nvos_render_views.py --scene=leaves
python .\gsd_nvos_use_sam2.py --scene=leaves
python gsd_nvos_obj_removal.py --scene=leaves --out --with_var
--> data/mid/gsd/leaves_crt_0.5_stride_1_radius_0_bb_0.0_var_0.2
python gsd_nvos_metrics.py --scene=leaves --with_var 

iou 0.9295360687404249
acc 0.9955921875

## orchids

iou 0.9015233320011142
acc 0.9833279859011506

iou 0.9071544196997939
acc 0.9842270833333333

python .\gsd_nvos_render_views.py --scene=orchids
python .\gsd_nvos_use_sam2.py --scene=orchids
python gsd_nvos_obj_removal.py --scene=orchids --out --with_var
--> data/mid/gsd/orchids_crt_0.5_stride_1_radius_0_bb_0.0_var_0.2
python gsd_nvos_metrics.py --scene=orchids --with_var 

iou 0.8969915370589198
acc 0.9826109375

iou 0.9015340145697487
acc 0.983357812

## trex

iou 0.7883547276429974
acc 0.971616289262618

iou 0.8677654318123825
acc 0.9821859375

python .\gsd_nvos_render_views.py --scene=trex
python .\gsd_nvos_use_sam2.py --scene=trex
python gsd_nvos_obj_removal.py --scene=trex --out --with_var
--> data/mid/gsd/trex_crt_0.5_stride_1_radius_0_bb_0.0_var_0.2
python gsd_nvos_metrics.py --scene=trex --with_var 

iou 0.8677654318123825
acc 0.9821859375

## Result 

Mean IOU: 

0.9669856401810455+0.9412250895399027+0.8220298794434462+0.957980224378991+0.9711842929750052+0.9295360687404249+0.9015340145697487+0.8677654318123825

7.35824064164095 / 8 = 0.919780080205119

Mean ACC: 

0.9821859375+0.983357812+0.9955921875+0.9946109375+0.9903984375+0.942534375+0.9963333333333333+0.9942786458333334

7.87929166616667 / 8 

0.984911458270834