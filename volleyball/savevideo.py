from viewresult import ViewOnly as vo
import os

dirpath = '/home/junkado/Desktop/Hisamitsu/3d-data/'
files = os.listdir(dirpath + 'Motioncsv/')

for file in files:
    if file.find('.CSV') == -1:
        continue
    view = vo(dirpath + 'Motioncsv/' + file, dim=3, remove_rows=slice(0,6), remove_cols=slice(0,2))
    view.view(dirpath + 'savedata/' + file.replace('.CSV', '.MP4'), saveonly=True)