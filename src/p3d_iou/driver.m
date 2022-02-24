name = 'p3d_CAR'
split = 'val'
class = 'car'

if exist('/scratch3')
    p3d_dir = '/scratch3/shubham/data/PASCAL3D+_release1.1';
else
    p3d_dir = '/data1/shubhtuls/cachedir/PASCAL3D+_release1.1';
end
eval_mean_iou(name, split, class, p3d_dir);
