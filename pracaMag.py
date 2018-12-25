import os
import time
import scipy.signal as sig
import scipy.io.wavfile as w
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np
import pickle
from stat import *
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class Spectro():
    def __init__(self, f, t, Sxx):
        self.f = f
        self.t = t
        self.Sxx = Sxx
    className = []


class WavData():
    def __init__(self, fs, data, className):
        self.fs = fs
        self.data = data
        self.className = className


class ClassInfo():
    maxDuration = 0
    minDuration = 0
    ClassSpectro = Spectro([], [], [])


class AcousticEvent():
    def __init__(self, Wav, duration, when, sampleDuration=22050):  # konstruktor klasy

        self.duration = duration  # czas trwania probki
        self.when = when  # kiedy sie wystapilo
        if len(Wav.data) < sampleDuration:  # dopisanie zerami zeby mozna porownywac spektrogramy
            length = sampleDuration - len(Wav.data)
            zeros = np.zeros(length)
            Wav.data = np.append(Wav.data, zeros)
        self.Wav = (Wav)                                             # dane wav zdarzenia
        # spektrogram zdarzenia
        self.f, self.t, self.Sxx = specCalculate(Wav.data, Wav.fs)
        self.Sxx = specNorm(self.Sxx)
        self.Spec = Spectro(self.f, self.t, self.Sxx)
        # nazwa klasy do jakiej zaklasyfikowano
        self.className = 'none'

    def plotSpecAE(self):
        plotSpectogram(self.Spec)

    def printInfo(self):
        print self.className, "created:", self.when
    
    



# ----------------------OPERACJE NA PLIKACH-------------------------------------


def readData(fileName):  # funkcja wczytuje plik wav ktory bedzie badany
    """ wczytuje plik wave 
        zwraca obiekt typu WavData"""
    try:
        fs, data = w.read(fileName)
    except IOError, (errno, strerror):
        print "ERROR (%s): %s" % (errno, strerror)
        data = None
        fs = 0

    # TODO zrobic zeby obiekt nosil nazwe klasy do ktroej sie zalicza
    return WavData(fs, data, 'ala')


def writeWavFile(Wav, filename):
    """zapisuje plik wave"""
    w.write(filename, Wav.fs, Wav.data)
    return


def readAllAudio():
    """wczytuje wszystkie pliki wave z katalogu
       w obiekcie wavArray i dodaje je do slownika
       dict[nazwa klasy] zwraca ten ten slownik"""
    dictAllData = {}
    dn = ".//"
    fl = os.listdir(dn)
    drs = [f for f in fl if not os.path.isfile(os.path.join(dn, f))]

    for path in drs:
        tmp = readAllAudioFromDir(path)
        if len(tmp) == 0:
            continue
        else:
            dictAllData[path] = tmp
    return dictAllData


def readAllAudioFromDir(dirName):  # TODO zamiana na mono jezeli jest stereo
    """wczytuje wszystkie pliki wave z wybranego katalogu
       zwraca wavArray = [data, fs]"""
    wavArray = []
    audioFileNames = []
    # dn = ".//"
    dn = dirName

    try:
        files = os.listdir(dn)
    except OSError, (errno, strerror):
        print "ERROR (%s): %s" % (errno, strerror)
        return

    # fls = [f for f in os.listdir(dn) if os.path.isfile(os.path.join(dn, f))]
    fls = []
    for f in files:
        if os.path.isfile(os.path.join(dn, f)):
            fls.append(os.path.join(dn, f))

    # print fls
    for fl in fls:
        if fl[-4:-1] + fl[-1] == ".wav":
            audioFileNames.append(fl)

    if len(audioFileNames) == 0:
        print "there is no audio files in directory"
    else:
        for fl in audioFileNames:
            fs, data = w.read(fl)
            if sum(data) == 0:  # jakby wczytalo pusty plik
                continue
            data = norm(data)  # XXX Normalizacja pliki dzwiekowego
            wavArray.append((fs, data))

    return wavArray


def writeSpecFile(f, t, Sxx, fileName):
    """zapisuje spektrogram do pliku"""
    A = Spectro(f, t, Sxx)
    output = open(fileName + '.spec', 'wb')
    pickle.dump(A, output)
    output.close()
    return


def writeAllEvents(eventDict):
    """zapisuje wszystkie zdarzenia dzwiekowe
       parametr to slownik z wavArray"""
    dir = 'Pliki//'
    counter = 0
    for event in eventDict:
        if os.path.isfile(eventDict[event].className + '.wav'):
            filename = eventDict[event].className + str(counter)
            counter += 1
        else:
            filename =  eventDict[event].when + eventDict[event].className
        writeWavFile(eventDict[event].Wav,dir + filename + '.wav')
    return


def readSpecFile(fileName):
    """wczytuje spektrogram z pliku 
    zwraca obiekt klasy Spectro"""
    fl = open(fileName, 'rb')  # TODO wyjatek nie ma pliku
    A = pickle.load(fl)
    fl.close()
    return A


def readAllSpecFiles():
    """wczytuje wszystkie pliki *.spec z katalogu
    i dodaje je do slownika. Klucz to nazwa pliku
    zwraca slownik"""

    dn = ".//"
    spectroDict = {}
    fl = os.listdir(dn)
    fls = [f for f in fl if os.path.isfile(os.path.join(dn, f))]
    specFileNames = []
    # print fls

    for fl in fls:
        if fl[-5:-1] + fl[-1] == ".spec":
            specFileNames.append(fl)

    if len(specFileNames) == 0:
        print "W katalogu nie ma plikow z roszerzeniem .spec"

    for fn in specFileNames:
        A = readSpecFile(fn)
        A.className = fn
        spectroDict[fn] = A
    return spectroDict


def pngfromSpec():
    """Tworzy pliki .png z plikow .spec"""
    dn = ".//"
    fl = os.listdir(dn)
    fls = [f for f in fl if os.path.isfile(os.path.join(dn, f))]
    specFileNames = []
    # print fls

    for fl in fls:
        if fl[-5:-1] + fl[-1] == ".spec":
            specFileNames.append(fl)

    if len(specFileNames) == 0:
        print "W katalogu nie ma plikow z roszerzeniem .spec"

    for fn in specFileNames:
        A = readSpecFile(fn)
        plotSpectogram(A, name=fn)
        plt.savefig(fn + '.png')

    return


# ----------------------OBLICZENIA-------------------------------------


def norm(audioFile):
    """normalizuje plik dzwiekowy"""

    m = max(abs(audioFile))
    try:
        normFactor = 0.99 / m
    except ZeroDivisionError:
        print "max value of audiofile is 0"
        return

    audioFile = audioFile * normFactor
    return audioFile


def specCalculate(fileName, fs):
    """obliczanie spektogramu
    zwraca f, t, Sxx"""

    windowSize = 1024
    overLap = 0.75 * windowSize
    f, t, Sxx = sig.spectrogram(fileName, fs=fs, window=np.hamming(windowSize), nperseg=windowSize,
                                noverlap=overLap, scaling='spectrum', mode='magnitude')
    Tmp = Spectro(f, t, Sxx)
    return f, t, Sxx  # TODO: Zamienic na klase Spectro



def createSampleSpectogram(wavArray, mode):
    """Tworzy spektrogram usredniony z plikow wave 
    znajdujacych sie w katalogu. Nazwa katalogu jest 
    nazwa pliku dodany jest jedynie .spec. 
    mode = add: dodaje wszystkie spektrogramy i normalizuje wynik
    mode = mid: oblicza wartosc srednia ze wszystkich spektrogramow"""
    # TODO Czy  pliki tej samej dlugosci
    spectroArray = []
    print "Spektogram jest tworzony..."

    # for i in range(1, len(wavArray)):
    #     if len(wavArray[i]) != len(wavArray[i - 1]):
    #         print "Files are not the same size."
    #         return  # FIXME musi cos zwracac zeby wiedziec czy zapisac plik czy nie

    if mode == 'mid':
        for wav in wavArray:
            tmpWav = norm(wav[1])
            # FIXME: Zamienic na Sperctro
            f, t, Sxx = specCalculate(tmpWav, wav[0])
            spectroArray.append(Spectro(f, t, Sxx))
        sumSxx = 0
        for Spec in spectroArray:
            sumSxx = sumSxx + Spec.Sxx
        midSxx = sumSxx / float(len(spectroArray))
    elif mode == 'add':
        for wav in wavArray:
            tmpWav = norm(wav[1])
            # FIXME: Zamienic na Sperctro
            f, t, Sxx = specCalculate(tmpWav, wav[0])
            spectroArray.append(Spectro(f, t, Sxx))
        sumSxx = 0
        for Spec in spectroArray:
            sumSxx = sumSxx + Spec.Sxx
        midSxx = sumSxx

    midSxx = specNorm(midSxx)

    # print float(len(spectroArray))
    Tmp = Spectro(f, t, midSxx)

    # return Tmp
    return f, t, midSxx  # FIXME: zamienic na spectro


def specNorm(Sxx):
    """normalizacja spektrogramu"""
    m = np.amax(abs(Sxx))
    normFactor = 0.9999 / m
    Sxx = Sxx * normFactor
    return Sxx


def zerosCount(Sxx):  # Oblicza ilosc liczb wiekszych od zera w tablicy
    """Liczy liczby wieksze od zera w tablicy"""
    count = 0
    for i in Sxx:
        for j in i:
            if j > 0:
                count += 1
    return count


def timeToSamples(time, fs):
    """zwraca nr probki jako parametr przyjmujac czas i czestotliwosc"""
    for i in range(len(time)):
        time[i] = time[i] * fs
    return time


def samplesToTime(samples, fs):
    """zwraca czas jako parametr przyjmujac np probli i czestotliwosc"""
    fs = float(fs)
    samples = float(samples)
    return float(samples / fs)


# Funkcja tworzy spektogramy usrednione dla wszystkich podkartalogow
def createAllSampleSpectogram(mode='add'):
    """Funkcja tworzy spektogramy usrednione 
    dla wszystkich podkartalogow pod warunkiem
     ze zawieraja pliki .wav. Zwraca slownik 
     w formie dictBase[klasa] = [fs, wavData]
    - czyli wavArray"""
    dictBase = readAllAudio()
    for klasa in dictBase:
        # print klasa
        f, t, Sxx = createSampleSpectogram(dictBase[klasa], mode)
        writeSpecFile(f, t, Sxx, klasa)
    return dictBase


def slidingWindow(specTested, specPattern, step=1, start=0):
    """Liczy najmniejsza roznice pomiedzy spektogramem specTested a specPattern przesuwajac okno i kolejno porownujac"""
    x_tested = len(specTested.t)
    y_tested = len(specTested.f)
    x_pattern = len(specPattern.t)
    y_pattern = len(specPattern.f)

    if y_tested != y_pattern:
        print 'spektogramy o roznej wysokosci'
        return

    zeros = np.zeros([y_tested, x_tested + 2 * x_pattern])
    offset = x_pattern
    zeros[:, offset: offset + x_tested] = specTested.Sxx

    sumResult = []
    avrResult = []

    for i in range(start, np.size(zeros, 1) - x_pattern, step):
        specTmp = zeros[:, i: i + x_pattern]
        # tmpSxx = np.sqrt((specTmp - specPattern.Sxx)**2)
        tmpSxx = specCompare(specTmp, specPattern.Sxx)
        sumResult.append(sum(sum(tmpSxx)))
        avrResult.append(np.mean(tmpSxx))

    x = np.zeros(x_tested)
    count = 0
    for i in range(offset, offset + x_tested):
        x[count] = avrResult[i]
        # x[count] = sumResult[i]
        count += 1

    # plt.plot(x, 'g')
    # plotWav(x)
    return (avrResult)


def calibrateWav(wav):
    noise = []
    f, t, Sxx = specCalculate(wav.data, wav.fs)
    noise.append(max(sum(Sxx)))
    tmp = max(noise)
    return tmp


def calibrateSpec(spec):
    noise = []
    noise.append(max(sum(spec)))
    result = max(noise)
    return result


def eventDetection(path):  # funkcja wykrywa zdarzenia akustyczne

    eventDict = {}
    Wavfile = readData(path)
    Wavfile.data = norm(Wavfile.data)
    TmpWav = WavData(
        Wavfile.fs, Wavfile.data[10025:10025+22050], Wavfile.className)
    # plotWav(Wavfile.data)
    noise = calibrateWav(TmpWav)
    
    f, t, Sxx = specCalculate(Wavfile.data, Wavfile.fs)
    # Sxx = specNorm(Sxx)

    step = 10
    z = abs(sum(Sxx[:, 0:-step] - Sxx[:, step-1:-1]))
    # z1 = (sum(Sxx[:,step-1:-1  ] - Sxx[:,0:-step ]))

    z = rms((z), 5)

    index = np.where(z > 3*np.mean(z))
    for i in range(len(z)):
        z[i] = 0
        pass
    for i in index[0]:
        z[i] = 1

    noiseFactor = -7.0175 * noise + 6.2105 
    if noise > 0.1:
        noiseFactor = 2
    print 'noisefactor =', noiseFactor
    x = sum(Sxx)
    print 'szum =', noise
    x = rms(x, 5)  # opcja
    for i in range(0, len(x)):
        if x[i] <= noiseFactor * noise: # ile razy wieksze od szumu 
            x[i] = 0
        else:
            x[i] = 0.001
    # x = z
    # plotWav(x)
    plt.plot(x, 'b')

    s = []  # tablica poczatkow i koncow zdarzen
    cff = int(float(len(Wavfile.data)) / float(len(x)))
    for i in range(1, len(x)):
        if x[i] != x[i-1]:
            s.append(i * cff)

    st = os.stat(path)
    fileCount = 0

    for i in range(0, len(s) - 1, 2):
        if fileCount < 10:
            name = 'none0' + str(fileCount)
        else:
            name = 'none' + str(fileCount)
        fileCount += 1
        duration = [samplesToTime(s[i], Wavfile.fs),
                    samplesToTime(s[i + 1], Wavfile.fs)]

        Tmp = WavData(Wavfile.fs, Wavfile.data[(
            s[i]):s[i + 1]], Wavfile.className)

        eTime = time.asctime(time.localtime(st[ST_MTIME] + duration[0]))
        eTmp = AcousticEvent(Tmp, duration, eTime)
        eventDict[name] = eTmp

    return eventDict


def rms(data, samplesCount):
    tmpData = []
    mid = 0.0
    for i in range(0, len(data) - samplesCount, int(samplesCount)):
        for j in range(0, samplesCount):
            mid += data[i + j]
        mid = mid / samplesCount
        for j in range(0, samplesCount):
            tmpData.append(mid)
        mid = 0

    return tmpData


def specCompare(A, B):

    result = np.sqrt((A - B)**2)

    return result


def classyfication(EventDict):
    spectroDict = readAllSpecFiles()

    for event in EventDict:
        minDistance = 100000
        for spec in spectroDict:
            distance = (slidingWindow(
                spectroDict[spec], EventDict[event].Spec)) # TODO zabezpieczyc jak nie ma plikow .spec
            if distance < minDistance:
                minDistance = distance
                EventDict[event].className = spec[0:-5]
        print distance, ":\t", EventDict[event].printInfo()

    return EventDict

# ----------------------WIZUALIZACJA-------------------------------------


def plotSpectogram(spec, name='spectogram'):

    f = spec.f
    t = spec.t
    Sxx = spec.Sxx
    # if spec.className != '':
    #     name = spec.className 
    plt.figure(figsize=(16, 8))
    # plt.pcolormesh(t, f, 20 * np.log10(Sxx))
    plt.pcolormesh(t, f, (Sxx))
    plt.xlabel('czas [s]')
    plt.ylabel('czestotliwosc [Hz]')
    plt.title(name)
    plt.ylim(0, 10000)
    # plt.xlim(0, 0.2)
    plt.colorbar()


def plotWav(fileName, name=''):
    plt.figure()
    plt.plot(fileName)
    plt.title(name)


# createAllSampleSpectogram()
# pngfromSpec()
dir = 'testy1.wav'
d = eventDetection(dir)
wav = readData(dir)
wav.data = norm(wav.data)

print len(d)
# d = classyfication(d)
# writeAllEvents(d)
f, t, Sxx = specCalculate(wav.data[0:5025], wav.fs)
A = Spectro(f, t, (Sxx))
f1, t1, Sxx1= specCalculate(wav.data, wav.fs)
test = Spectro(f1, t1, (Sxx1))

plotSpectogram((test))
# specs = readAllSpecFiles()
# wav.data = 20 * np.log10(wav.data)# stri = 'none00'
# f, t, Sxx = specCalculate(wav.data, wav.fs)
# A = Spectro(f, t, Sxx)
# x = slidingWindow( d[stri].Spec  ,specs['polsekundy.spec'] )
# for sp in specs:
x = slidingWindow(test, A)
z = np.zeros(len(x))
for i in range(len(x) - 1):
    z[i] = x[i + 1] - x[i]

# plt.plot(z)

# plotWav(x)
# x = slidingWindow(A, specs['drzwi.spec'])
# x = slidingWindow( specs['polsekundy.spec'],specs['klik.spec'] )

# plotSpectogram(d[stri].Spec, stri)
# plotSpectogram(specs['klik.spec'], 'klik')
# plotWav(sum(specs['drzwi.spec'].Sxx))
# plt.plot(wav.data, 'b')
# plt.plot((x), 'r')
# plt.plot(x, 'g')

plt.show()
# print min(x)
