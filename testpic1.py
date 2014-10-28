import cv2
import numpy as np
import nltk
import os
from glob import glob
from matplotlib import pyplot as plt

def mode(list):
	d = {}
	for elm in list:
		try:
			d[elm] += 1
		except(KeyError):
			d[elm] = 1
	
	keys = d.keys()
	max = d[keys[0]]
	
	for key in keys[1:]:
		if d[key] > max:
			max = d[key]
	max_k = []
	for key in keys:
		if d[key] == max:
			max_k.append(key),
	return max_k,max



PS = nltk.stem.porter.PorterStemmer()
stem = PS.stem

sift = cv2.SIFT()
bf = cv2.BFMatcher()

d = {}
flist = glob("*.txt")
for fname in flist:
	imgname = '.'.join(fname.split('.')[:-1])+".jpg"
	print fname
	print imgname
	print
	
	with open(fname) as fp:
		text = fp.read()
	for char in '."?!:\t\n':
		text.replace(char,' ')
	
	words = [stem(word.lower()) for word in text.split()]
	
	img = cv2.imread(imgname,0)
	kp, des = sift.detectAndCompute(img, None)
	
	for word in words:
		if word not in d:
			d[word] = [[],[],[],[]]
			for i,pt in enumerate(des):
				d[word][0].append(pt)
				d[word][1].append(1)
				d[word][2].append(imgname)
				d[word][3].append(kp[i])
			print "NEW:", word
		else:
			des0 = np.array(d[word][0])
			matches = bf.knnMatch(des0,des, k=2)
			
			good = []
			for m,n in matches:
				if m.distance < 0.75*n.distance:
					good.append(m)
			
			print word
			
			for gw in good:
				d[word][1][gw.queryIdx]+=1.2
			
			for k in range(len(d[word][1])):
				d[word][1][k] -= 0.2
			
			repetitions = [k.trainIdx for k in good]
			for i in xrange(len(des)):
				if i not in repetitions:
					d[word][0].append(des[i])
					d[word][1].append(1)
					d[word][2].append(imgname)
					d[word][3].append(kp[i])

for word in d:
	mxlist = sorted(d[word][1])[-10:]
	milist = [d[word][1].index(k) for k in mxlist]
	
	mg,_ = mode([d[word][2][k] for k in milist])
	mg = mg[0]
	
	mflist = []
	for k in milist:
		if d[word][2][k] == mg:
			mflist.append(k)
	
	mi = d[word][1].index(max(d[word][1]))
	print word
	
	img = cv2.imread(mg,0)
	img2 = cv2.drawKeypoints(img, [d[word][3][k] for k in mflist])
	plt.imshow(img2),plt.show()