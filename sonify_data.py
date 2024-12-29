import numpy as np
from scipy.io import wavfile
import ehtim as eh
from pydub import AudioSegment
import matplotlib.pyplot as plt

def sonify(obs,outname,dt=10./60.,plot_coverage=False):
	"""
	Given an observation (in this case, an eht-imaging obsdata object),
	sonify it by converting visibilities to tones.
	dt describes the duration of each observation;
	it should correspond roughly to scan length in hours, 
	or some multiple thereof.
	"""
	obs.add_scans()
	obs = obs.avg_coherent(0.,scan_avg=True)
	duration_hrs = obs.data['time'][-1]-obs.data['time'][0]

	data = obs.unpack(['uvdist','amp','phase','time','u','v'],debias=True)

	uvd = obs.unpack(['u','v'],conj=True)
	if plot_coverage:
		plt.plot(uvd['u']/1e9,uvd['v']/1e9,'.')
		plt.xlim((35,-35))
		plt.ylim((-35,35))
		plt.xlabel('u')
		plt.ylabel('v')
		plt.gca().set_aspect('equal')
		plt.title(outname)
		plt.savefig(outname+'.png',bbox_inches='tight')
		plt.close('all')

	#human hearing range is roughly 20 Hz - 20kHz
	sampleRate = 44100
	eps = 1
	duration = int(np.ceil(duration_hrs)*eps)
	dt = eps*dt
	t = np.linspace(0, duration, sampleRate * duration)  #  Produces a 5 second Audio-File
	uleft = np.zeros_like(t)
	vright = np.zeros_like(t)
	full = np.zeros_like(t)

	freq_scale = 1/2*1e-9 * 1e2 #convert lambda to Glambda to kHz to 100 Hz

	for i in range(len(data)):
		uv_angle = np.arctan2(data[i]['u'],data[i]['v'])
		amp = data[i]['amp']
		freq = data[i]['uvdist']*freq_scale
		phase = data[i]['phase']
		time = eps*(data[i]['time'] - data[0]['time'])
		uleft = uleft+amp*np.sin(freq*t*2*np.pi+phase)*np.exp(-(t-time)**2 / dt**2)*np.cos(uv_angle)
		vright = vright+amp*np.sin(freq*t*2*np.pi+phase)*np.exp(-(t-time)**2 / dt**2)*np.sin(uv_angle)
		full = full+amp*np.sin(freq*t*2*np.pi+phase)*np.exp(-(t-time)**2 / dt**2)

	muleft = np.max(np.abs(uleft))
	mvright = np.max(np.abs(vright))
	mfull = np.max(np.abs(full))
	# print("m", m)

	maxint16 = np.iinfo(np.int16).max  # == 2**15-1

	uleft = maxint16 * uleft / muleft

	uleft = uleft.astype(np.int16) 

	vright = maxint16 * vright / mvright

	vright = vright.astype(np.int16) 

	full = maxint16 * full/mfull
	full = full.astype(np.int16)

	wavfile.write(outname+'_uleft.wav', sampleRate, uleft)
	wavfile.write(outname+'_vright.wav', sampleRate, vright)
	wavfile.write(outname+'_full.wav', sampleRate, full)


	left_channel = AudioSegment.from_wav(outname+'_uleft.wav')
	right_channel = AudioSegment.from_wav(outname+"_vright.wav")

	stereo_sound = AudioSegment.from_mono_audiosegments(left_channel, right_channel)
	stereo_sound.export(outname+'_binaural.mp3',format='mp3')

#An example with public EHT data:
# obs = eh.obsdata.load_uvfits('./m87_zblcal+selfcal/hops_lo_3597_M87+zbl-dtcal_selfcal.uvfits')
# sonify(obs,'EHT2017_3597_lowband',dt=20/60)
