import numpy as np
from pypher.pypher import psf2otf
import PIL.Image as Image

def L0Smoothing(Im, lambd = 2e-2, kappa = 2.0):
	# L0 Smoothing
	# Input:
	#   Im: Input UINT8 image, both grayscale and color images are acceptable.
	#   lambd: Smoothing parameter controlling the degree of smooth. (See [1])
	#          Typically it is within the range [1e-3, 1e-1], 2e-2 by default.
	#   kappa: Parameter that controls the rate. (See [1])
	#          Small kappa results in more iteratioins and with sharper edges.
	#          We select kappa in (1, 2].
	#          kappa = 2 is suggested for natural images.

	# Example:
	#   Im = imread('test.png')
	#   S = L0Smoothing(Im, 2e-2, 2.0)
	#   imshow(S)

	S = Im.astype(np.float64) / 255.0
	betamax = 1e5
	fx = np.array([[-1, 1]])
	fy = np.array([[-1], [1]])
	N, M, D = S.shape
	sizeI2D = np.array([N, M])
	otfFx = psf2otf(fx, sizeI2D)
	otfFy = psf2otf(fy, sizeI2D)
	Normin1 = np.fft.fft2(S.T).T
	Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
	if D > 1:
		D2 = np.zeros((N, M, D), dtype=np.double)
		for i in range(D):
			D2[:, :, i] = Denormin2
		Denormin2 = D2
	beta = lambd * 2
	while beta < betamax:
		Denormin = 1 + beta * Denormin2
		# Referenced from L-Dreams's blog
		# h-v subproblem
		h1 = np.diff(S, 1, 1)
		h2 = np.reshape(S[:, 0], (N, 1, 3)) - np.reshape(S[:, -1], (N, 1, 3))
		h = np.hstack((h1, h2))
		v1 = np.diff(S, 1, 0)
		v2 = np.reshape(S[0, :], (1, M, 3)) - np.reshape(S[-1, :], (1, M, 3))
		v = np.vstack((v1, v2))
		if D == 1:
			t = (h ** 2 + v ** 2) < lambd / beta
		else:
			t = np.sum((h ** 2 + v ** 2), 2) < lambd / beta
			t1 = np.zeros((N, M, D), dtype=bool)
			for i in range(D):
				t1[:, :, i] = t
			t = t1
		h[t] = 0
		v[t] = 0
		# S subproblem
		Normin2 = np.hstack((np.reshape(h[:, -1], (N, 1, 3)) - np.reshape(h[:, 0], (N, 1, 3)), -np.diff(h, 1, 1)))
		Normin2 = Normin2 + np.vstack(
			(np.reshape(v[-1, :], (1, M, 3)) - np.reshape(v[0, :], (1, M, 3)), -np.diff(v, 1, 0)))
		FS = (Normin1 + beta * np.fft.fft2(Normin2.T).T) / Denormin
		S = np.real(np.fft.ifft2(FS.T).T)
		beta *= kappa
	return S

if __name__ == '__main__':
	#input image
	im = np.array(Image.open('pflower.jpg'))
	print("Image Loaded.")
	#L0 Smoothing
	print("Image Processing.")
	S = L0Smoothing(im, 2e-2, 2.0)
	#save image
	print("Image Saving.")
	S = Image.fromarray(np.uint8(S*255))
	S.save('pflower_L0Smoothing.jpg')
	print("Done.")




