import os

filename = 'fold_frontal_0_data.txt'
savepath = 'list/' + filename[0:-9] + '.txt'
print savepath

content = ''
with open(filename,'r') as f:
    lines = f.readlines()
for line in lines:
    s = line.strip().split('\t')
    if s[4]!='f' and s[4]!='m':
        continue
    label = 0 if s[4]=='f' else 1

    imgpath = 'aligned/' + s[0] + '/landmark_aligned_face.' + s[2] + '.' + s[1]
    if os.path.exists(imgpath)==False:
        print imgpath
        continue

    line_save = s[0] + '/landmark_aligned_face.' + s[2] + '.' + s[1] + ' ' + str(label) + '\n'
    content = content + line_save
with open(savepath,'w') as f:
    f.write(content)