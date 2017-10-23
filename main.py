import numpy as np
from PIL import Image
from itertools import izip
from scipy.ndimage import sobel
from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
from scipy.fftpack import fftshift as __fftshift
from scipy.ndimage.filters import convolve as __convolve
from scipy.ndimage.filters import correlate as __correlate
from scipy.ndimage.filters import gaussian_filter as __gaussian_filter

def si(image):
	"""Computes the Spatial Information (SI) of an image.

    mean, rms, stddev = si(image)

    Parameters
    ----------
    image    : original image data (grayscale).

    Return
    ----------
    mean    : SI Mean
    rms     : SI Root-Mean-Square
    stddev  : SI Standard Deviation
    
    More info: http://ieeexplore.ieee.org/document/6603194/
    """
	h_sobel = sobel(image, 0)
	v_sobel = sobel(image, 1)
	si_sobel = np.hypot(h_sobel, v_sobel)

	sr = []

	for i in range(0,image.shape[0]):
		for j in range(0, image.shape[1]):
			sr.append(si_sobel[i][j])

	return np.mean(sr), np.sqrt(np.mean(np.power(sr, 2))), np.std(sr)

def mse(reference, query):
	"""Computes the Mean Square Error (MSE) of two images.

	value = mse(reference, query)

	Parameters
	----------
	reference: original image data.
	query    : modified image data to be compared.

	Return
	----------
	value    : MSE value
	
	More info: https://bitbucket.org/kuraiev/pymetrikz
	"""
	(ref, que) = (reference.astype('double'), query.astype('double'))
	diff = ref - que
	square = (diff ** 2)
	mean = square.mean()
	return mean

def psnr(reference, query, normal=255):
	"""Computes the Peak Signal-to-Noise-Ratio (PSNR).

	value = psnr(reference, query, normalization=255)

	Parameters
	----------
	reference: original image data.
	query    : modified image data to be compared.
	normal   : normalization value (255 for 8-bit image

	Return
	----------
	value    : PSNR value
	
	More info: https://bitbucket.org/kuraiev/pymetrikz
	"""
	normalization = float(normal)
	msev = mse(reference, query)
	if msev != 0:
		value = 10.0 * np.log10(normalization * normalization / msev)
	else:
		value = float("inf")
	return value

def ssim(reference, query):
	"""Computes the Structural SIMilarity Index (SSIM).

	value = ssim(reference, query)

	Parameters
	----------
	reference: original image data.
	query    : modified image data to be compared.

	Return
	----------
	value    : SSIM value
	
	More info: https://bitbucket.org/kuraiev/pymetrikz
	"""
	def __get_kernels():
		k1, k2, l = (0.01, 0.03, 255.0)
		kern1, kern2 = map(lambda x: (x * l) ** 2, (k1, k2))
		return kern1, kern2

	def __get_mus(i1, i2):
		mu1, mu2 = map(lambda x: __gaussian_filter(x, 1.5), (i1, i2))
		m1m1, m2m2, m1m2 = (mu1 * mu1, mu2 * mu2, mu1 * mu2)
		return m1m1, m2m2, m1m2

	def __get_sigmas(i1, i2, delta1, delta2, delta12):
		f1 = __gaussian_filter(i1 * i1, 1.5) - delta1
		f2 = __gaussian_filter(i2 * i2, 1.5) - delta2
		f12 = __gaussian_filter(i1 * i2, 1.5) - delta12
		return f1, f2, f12

	def __get_positive_ssimap(C1, C2, m1m2, mu11, mu22, s12, s1s1, s2s2):
		num = (2 * m1m2 + C1) * (2 * s12 + C2)
		den = (mu11 + mu22 + C1) * (s1s1 + s2s2 + C2)
		return num / den

	def __get_negative_ssimap(C1, C2, m1m2, m11, m22, s12, s1s1, s2s2):
		(num1, num2) = (2.0 * m1m2 + C1, 2.0 * s12 + C2)
		(den1, den2) = (m11 + m22 + C1, s1s1 + s2s2 + C2)
		ssim_map = np.ones(img1.shape)
		indx = (den1 * den2 > 0)
		ssim_map[indx] = (num1[indx] * num2[indx]) / (den1[indx] * den2[indx])
		indx = np.bitwise_and(den1 != 0, den2 == 0)
		ssim_map[indx] = num1[indx] / den1[indx]
		return ssim_map

	(img1, img2) = (reference.astype('double'), query.astype('double'))
	(m1m1, m2m2, m1m2) = __get_mus(img1, img2)
	(s1, s2, s12) = __get_sigmas(img1, img2, m1m1, m2m2, m1m2)
	(C1, C2) = __get_kernels()
	if C1 > 0 and C2 > 0:
		ssim_map = __get_positive_ssimap(C1, C2, m1m2, m1m1, m2m2, s12, s1, s2)
	else:
		ssim_map = __get_negative_ssimap(C1, C2, m1m2, m1m1, m2m2, s12, s1, s2)
	ssim_value = ssim_map.mean()
	return ssim_value

def pwssim(reference, ref_gs, query):
	"""Computes the Structural SIMilarity Index w/ Perceptual Weighting (PWSSIM).

	value = pwssim(reference, ref_gs, query)

	Parameters
	----------
	reference: original image data.
	ref_gs   : original image data (grayscale).
	query    : modified image data to be compared.

	Return
	----------
	value    : PWSSIM value
	
	More info: http://ieeexplore.ieee.org/document/7069106/
	"""
	si_value = [] # SI values for each block
	ssim_value = [] # SSIM values for each block
	pwssim_num = [] # Auxiliary variable
	pwssim_den = [] # Auxiliary variable
	for row_start, row_end in zip(range(0,reference.shape[0] + 1,8), range(8,reference.shape[0] + 1,8)):
		for col_start, col_end in zip(range(0,reference.shape[1] + 1, 8), range(8, reference.shape[1] + 1,8)):
			si_value.append(si(ref_gs[row_start:row_end,col_start:col_end]))
			ssim_value.append(ssim(reference[row_start:row_end,col_start:col_end], query[row_start:row_end,col_start:col_end]))
	for i in range(0,len(si_value)):
		pwssim_num.append(si_value[i][2] * ssim_value[i])
		pwssim_den.append(si_value[i][2])
	if sum(pwssim_den) == 0:
		return np.nan
	else:
		pwssim = sum(pwssim_num)/sum(pwssim_den)
		return pwssim

def set_lsb(value, bit, lsb):
	""" It assists in setting the LSB of the value passed in
		based on the binary to be hidden """
	mask = list(value)
	j = 0
	for i in range(7, 7-lsb, -1):
		if bit[j] == '0':
			mask[i] = '0'
		else:
			mask[i] = '1'
		j += 1
	value = ''.join(mask)
	subs = '0b'
	subs += value
	value = int(subs, 2)
	return value

def get_pixel_pairs(iterable):
	""" It takes in the pixel list and returns the pairs back. """
	a = iter(iterable)
	return izip(a, a)

def get_lsb(value, lsb):
	""" It'll take in an RGBA value and use a bit mask to get
		the LSB value and return it in a str format. """
	mask = list(value)
	bits = ""
	for i in range(7, 7-lsb, -1):
		if mask[i] == '0':
			bits += '0'
		else:
			bits += '1'
	return bits

def hide_message(carrier, message, output, num_bits):
	if len(message) == 0:
		raise ValueError('There is no message to hide.' )
	# Adds the value 0x00 to the end of the string to indicate
	# that we've reached the end of the hidden text
	message += chr(0)
	while (1.0 * len(message) / num_bits) != int(len(message) / num_bits):
		message += chr(0)
	print"The hidden message contains", len(message), "characters."
	c_image = Image.open(carrier)
	c_image = c_image.convert('RGBA')
	out = Image.new(c_image.mode, c_image.size)

	# All the pixel data from the carrier image
	pixel_list = list(c_image.getdata())
	# It'll hold all the new pixel values for the combined carrier and message image
	new_array = pixel_list

	if (len(message) - 1) > (len(pixel_list) * num_bits / 2):
		raise ValueError('Too much data for carrier image.')

	# Looping over each character in the message
	for i in range(0, (len(message) / num_bits)):
		cb = ""
		for j in range((i * num_bits), (i * num_bits + num_bits)):
			# Converting each characther to an int.
			char_int = ord(message[j])
			# Convert that int into a binary string
			# zfiil ensures that it's 8 character long
			cb += str(bin(char_int))[2:].zfill(8)

		pix1 = pixel_list[i * 2]
		pix2 = pixel_list[(i * 2) + 1]
		newpix1 = []
		newpix2 = []

		# It iterates 4 times and call set_lsb method.
		for k in range(0,4):
			pixel1 = str(bin(pix1[k]))[2:].zfill(8)
			pixel2 = str(bin(pix2[k]))[2:].zfill(8)
			newpix1.append(set_lsb(pixel1, cb[k * num_bits:(k+1) * num_bits], num_bits))
			newpix2.append(set_lsb(pixel2, cb[(k+4) * num_bits:(k+4) * num_bits + num_bits], num_bits))

		new_array[i*2] = (tuple(newpix1))
		new_array[i*2+1] = (tuple(newpix2))

	out.putdata(new_array)
	out.save(output)
	print "Output image saved as", output

def extract_message(steg_image, num_bits):
	# Image object from the filename passed
	s_image = Image.open(steg_image)
	# Array of pixels from the image data
	pixel_list = list(s_image.getdata())
	message = ""
	message_byte = ""

	# Iterating over all of the pixel pairs returned using get_pixel_pairs
	for pix1, pix2 in get_pixel_pairs(pixel_list):
		for p in pix1:
			pa = str(bin(p))[2:].zfill(8)
			message_byte += get_lsb(pa, num_bits)
			
		for p in pix2:
			pa = str(bin(p))[2:].zfill(8)
			message_byte += get_lsb(pa, num_bits)
		
		if message_byte[len(message_byte)-8:] == "00000000":
			break
	for i in range(0, len(message_byte), 8):
		message += chr(int(message_byte[i:i + 8], 2))
	message += chr(0)
	return message
