export PYTHONPATH=`pwd`
#exp darkframe
python scripts/save_cam_image.py -iso 3100 -exps 1/50,1/25,1/10,1/8,1/5,1/3,0.4,0.5,0.6,0.8,1.0  -si 0 #21
#iso darkframe
python scripts/save_cam_image.py -iso 200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100 -exps 1/50  -si 22 #81
python scripts/prepare_cam_raw.py

