# the input to this file should be the directory of your data (i.e. 'house' or 'library')
import numpy as np
import cv2
import cv
import skimage.io as skio
import scipy.io as spio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def compute_T(pts):
	mu = np.mean(pts,axis=0)
	sigma = 0.
	for i in range(len(pts)):
		sigma += (pts[i][0]-mu[0])**2 + (pts[i][1] - mu[1])**2
	sigma /= (2.*len(pts))
	sigma = np.sqrt(sigma)
	T = np.array([[1./sigma,0.,-mu[0]/sigma],[0.,1./sigma,-mu[1]/sigma],[0.,0.,1.]])
	return T


class Reconstruct:
	def __init__(self,name):
		self.data_dir = "../data/" + name+ "/"
		self.im1 = skio.imread(self.data_dir + name + "1.jpg")
		self.im2 = skio.imread(self.data_dir + name + "2.jpg")
		self.K1 = spio.loadmat(self.data_dir + name + "1_K.mat")['K']
		self.K2 = spio.loadmat(self.data_dir + name + "2_K.mat")['K']
		self.matches = [x.rstrip().split() for x in open(self.data_dir + name + "_matches.txt")]
		for i in range(len(self.matches)):
			for j in range(len(self.matches[i])):
				self.matches[i][j] = float(self.matches[i][j])
		self.matches = np.array(self.matches)

	def compute_test_fund(self):
		im1_pts = self.matches[:,:2]
		n = im1_pts.shape[0]
		im2_pts = self.matches[:,2:]		
		F,mask = cv2.findFundamentalMat(np.float32(im1_pts),np.float32(im2_pts),cv2.FM_8POINT)
		res = 0.
		for i in range(n):
			vec1 = np.array([im1_pts[i][0],im1_pts[i][1],1.])
			vec2 = np.array([im2_pts[i][0],im2_pts[i][1],1.])
			d12 = np.abs(np.dot(vec1.T,np.dot(F,vec2)))/np.linalg.norm(np.dot(F,vec2),2)
			d21 = np.abs(np.dot(vec2.T,np.dot(F,vec1)))/np.linalg.norm(np.dot(F,vec1),2)
			res += (d12**2) + (d21**2)
		res /= (2.*n)
		self.F = F
		return F,res

	def fundamental_matrix(self):
		im1_pts = self.matches[:,:2]
		n = im1_pts.shape[0]
		im2_pts = self.matches[:,2:]
		T1 = compute_T(im1_pts)
		T2 = compute_T(im2_pts)
		im1_pts = np.append(im1_pts,np.ones((n,1)),axis=1).T
		im2_pts = np.append(im2_pts,np.ones((n,1)),axis=1).T
		im1_pts = np.dot(T1,im1_pts).T[:,:2]
		im2_pts = np.dot(T2,im2_pts).T[:,:2]
		A = []
		for i in range(n):
			x1 = im1_pts[i][0]
			y1 = im1_pts[i][1]
			x2 = im2_pts[i][0]
			y2 = im2_pts[i][1]
			arr = np.array([x1*x2,y1*x2,x2,x1*y2,y1*y2, y2, x1, y1, 1.])
			A.append(arr)
		A = np.array(A)
		u,s,v = np.linalg.svd(A)
		f = v.T[:,-1]
		f = f.reshape(3,3)
		u,s,v = np.linalg.svd(f)
		s[-1]=0.
		F = np.dot(u,np.dot(np.diag(s),v))
		F = np.dot(T2.T,np.dot(F,T1))
		res = 0.
		for j in range(n):
			vec1 = np.array([im1_pts[j][0],im1_pts[j][1],1.])
			vec2 = np.array([im2_pts[j][0],im2_pts[j][1],1.])
			d12 = np.abs(np.dot(vec1.T,np.dot(F,vec2)))/np.linalg.norm(np.dot(F,vec2),2)
			d21 = np.abs(np.dot(vec2.T,np.dot(F,vec1)))/np.linalg.norm(np.dot(F,vec1),2)
			res += (d12**2) + (d21**2)
		res /= (2.*n)
		self.F = F
		return F,res

	def find_rotation_translation(self):
		self.E = np.dot(self.K2.T,np.dot(self.F,self.K1))
		u,s,v = np.linalg.svd(self.E)
		t = u[:,2]
		rot_90 =  np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
		rot_neg90 = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
		R = np.dot(u,np.dot(rot_90.T,v))
		R_neg = np.dot(u,np.dot(rot_neg90.T,v))
		self.pos_t = [t,-1.*t]
		self.pos_R = [R,-1.*R,R_neg,-1.*R_neg]
		return self.pos_t,self.pos_R


	def find_3d_points(self):
		B1 = np.append(np.identity(3),np.zeros((3,1)),axis=1)
		self.P1 = np.dot(self.K1,B1)
		num_points = np.ones((len(self.pos_t),len(self.pos_R)))
		rec_errs = np.ones((len(self.pos_t),len(self.pos_R)))

		def map_3d(p1,p2):
			im1_pts = self.matches[:,:2]
			im2_pts = self.matches[:,2:]
			camera_mats = [p1,p2]
			points = []
			for i in range(im1_pts.shape[0]):
				A = []
				for k in range(2):
					P = camera_mats[k]
					if k == 0:
						x,y = im1_pts[i]
					else:
						x,y = im2_pts[i]
					r1 = [P[2][0]*x - P[0][0], P[2][1]*x - P[0][1], P[2][2]*x - P[0][2],P[2][3]*x - P[0][3]]
					r2 = [P[2][0]*y - P[1][0], P[2][1]*y - P[1][1], P[2][2]*y - P[1][2],P[2][3]*y - P[1][3]]
					A.append(r1)
					A.append(r2)
				A = np.array(A)
				u,s,v = np.linalg.svd(A)
				pt = v.T[:,-1]
				pt /= pt[3]
				points.append(pt[:3])
			points = np.array(points)
			return points

		def rec_error(points,P1,P2):
			rec_err = 0.
			n = points.shape[0]
			im1_pts = self.matches[:,:2]
			im2_pts = self.matches[:,2:]
			X = np.append(points,np.ones(n).reshape(n,1),axis=1)
			proj1 = np.dot(P1,X.T)
			proj1 /= proj1[2,:]
			proj2 = np.dot(P2,X.T)
			proj2 /= proj2[2,:]
			err1 = (proj1.T[:,:2]-im1_pts)**2
			err2 = (proj2.T[:,:2]-im2_pts)**2
			mse1 = np.sqrt(err1.sum(axis=1)).mean()
  			mse2 = np.sqrt(err2.sum(axis=1)).mean()
			return mse1 + mse2
			 
		for i in range(num_points.shape[0]):
			t = self.pos_t[i]
			for j in range(num_points.shape[1]):
				R = self.pos_R[j]
				B2 = np.append(R,t.reshape(3,1),axis=1)
				P2 = np.dot(self.K2,B2)
				points_3d = map_3d(self.P1,P2)
				rec_errs[i][j] = rec_error(points_3d,self.P1,P2)
				Z1 = points_3d[:,2]
				Z2 = np.dot(R[2,:],points_3d.T)  + t[2]
				Z2 = Z2.T
				cond = Z1 > 0 
				cond2 = Z2 > 0 
				num_points[i][j] = np.sum(cond & cond2)

		ind = np.argmax(num_points)
		self.t = self.pos_t[ind / num_points.shape[1]]
		self.R = self.pos_R[ind % num_points.shape[1]]
		B2 = np.append(self.R,self.t.reshape(3,1),axis=1)
		self.P2 = np.dot(self.K2,B2)
		self.points = map_3d(self.P1,self.P2)
		rec_err = rec_error(self.points,self.P1,self.P2)
		return self.points,rec_err


	def plot_3d(self):
		# plot thte 3d points
		center1 = np.array([0,0,0]).reshape(1,3)
		center2 = -1.*np.dot(np.linalg.inv(self.R),self.t.reshape(3,1)).reshape(1,3)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(self.points[:,0],self.points[:,1],self.points[:,2],c='r',marker='o')
		ax.scatter(center1[:,0],center1[:,1],center1[:,2],c='b',marker='o')
		ax.scatter(center2[:,0],center2[:,1],center2[:,2],c='b',marker='o')
		plt.show()


if __name__=="__main__":
	library = Reconstruct("library")
	print "library: " 
	F,res =  library.compute_test_fund()
	post, posR = library.find_rotation_translation()
	points,err = library.find_3d_points()
	library.plot_3d()

	print "house: "
	house = Reconstruct("house")
	F,res =  house.compute_test_fund()
	post, posR = house.find_rotation_translation()
	points,err = house.find_3d_points()
	house.plot_3d()


