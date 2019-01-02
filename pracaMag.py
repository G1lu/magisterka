#!/usr/bin/env python3
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
import wave
import sounddevice as sd
import queue 
from threading import Thread


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

eventCount = 0
class AcousticEvent():
    def __init__(self, Wav, duration, when, sampleDuration=22050):  # konstruktor klasy
        global eventCount 
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
        self.className = 'none' + str(eventCount)
        eventCount += 1

    def plotSpecAE(self):
        plotSpectogram(self.Spec)

    def printInfo(self):
        print (self.className, "created:", self.when)
    
    



# ----------------------OPERACJE NA PLIKACH-------------------------------------


def readData(fileName):  # funkcja wczytuje plik wav ktory bedzie badany
    """ wczytuje plik wave 
        zwraca obiekt typu WavData"""
    try:
        fs, data = w.read(fileName)
    except IOError as err:
        print("ERROR (%s): %s" % format(err))
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
    except OSError as err:
        print ("ERROR (%s): %s" % format(err))
        return

    # fls = [f for f in os.listdir(dn) if os.path.isfile(os.path.join(dn, f))]
    fls = []
    for f in files:
        if os.path.isfile(os.path.join(dn, f)):
            fls.append(os.path.join(dn, f))

 
    for fl in fls:
        if fl[-4:-1] + fl[-1] == ".wav":
            audioFileNames.append(fl)

    if len(audioFileNames) == 0:
        print ("there is no audio files in directory")
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
    # fileName = 'Pliki//' + fileName 
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
            # filename =  eventDict[event].when + eventDict[event].className
            filename = eventDict[event].className
            print(filename)
            counter += 1
            
        writeWavFile(eventDict[event].Wav,dir + filename + str(counter) + '.wav')
    return


def readSpecFile(fileName):
    """wczytuje spektrogram z pliku 
    zwraca obiekt klasy Spectro"""
    fl = open( fileName, 'rb')  # TODO wyjatek nie ma pliku
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


    for fl in fls:
        if fl[-5:-1] + fl[-1] == ".spec":
            specFileNames.append(fl)

    if len(specFileNames) == 0:
        print ("W katalogu nie ma plikow z roszerzeniem .spec")

    for fn in specFileNames:
        A = readSpecFile(fn)
        A.className = fn
        spectroDict[fn] = A
    return spectroDict


def pngfromSpec(dn='.//'):
    """Tworzy pliki .png z plikow .spec"""
    
    fl = os.listdir(dn)
    fls = [f for f in fl if os.path.isfile(os.path.join(dn, f))]
    specFileNames = []
    

    for fl in fls:
        if fl[-5:-1] + fl[-1] == ".spec":
            specFileNames.append(fl)

    if len(specFileNames) == 0:
        print ("W katalogu nie ma plikow z roszerzeniem .spec")

    for fn in specFileNames:
        A = readSpecFile(fn)
        plotSpectogram(A, name=fn)
        plt.savefig(fn + '.png')

    return


# ----------------------OBLICZENIA-------------------------------------


def norm(audioFile):
    """normalizuje plik dzwiekowy"""

    m = max(abs(audioFile))
    if m == 0:
        normFactor = 1
    else:
        try:
            normFactor = 0.99 / m
        except ZeroDivisionError:
            print ("max value of audiofile is 0")
        
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
    print ("Spektogram jest tworzony...")

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


    Tmp = Spectro(f, t, midSxx)

    # return Tmp
    return f, t, midSxx  # FIXME: zamienic na spectro


def specNorm(Sxx):
    """normalizacja spektrogramu"""
    m = np.amax(abs(Sxx))
    if m == 0:
        normFactor = 1
    else:
        try:
            normFactor = 0.9999 / m        
        except ZeroDivisionError as err:
            print('ERROR', err)
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
        f, t, Sxx = createSampleSpectogram(dictBase[klasa], mode)
        writeSpecFile(f, t, Sxx,klasa)
    return dictBase


def slidingWindow(specTested, specPattern, step=1, start=0):
    """Liczy najmniejsza roznice pomiedzy spektogramem specTested a specPattern przesuwajac okno i kolejno porownujac"""
    x_tested = len(specTested.t)
    y_tested = len(specTested.f)
    x_pattern = len(specPattern.t)
    y_pattern = len(specPattern.f)

    if y_tested != y_pattern:
        print ('spektogramy o roznej wysokosci')
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
        for j in range(step):
            sumResult.append(sum(sum(tmpSxx)))
            avrResult.append(np.mean(tmpSxx))

    x = np.zeros(x_tested)
    count = 0
    for i in range(offset, offset + x_tested):
        x[count] = avrResult[i]
        # x[count] = sumResult[i]
        count += 1
    x = norm(x)
    plt.plot(x, 'g.')
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


def eventDetection(path, mode='object', fs=44100):  # funkcja wykrywa zdarzenia akustyczne

    eventDict = {}
    if mode == 'path':
        Wavfile = readData(path) # czyta z pliku 
        st = os.stat(path) # dla pliku
    elif mode == 'object':
        Wavfile = WavData(fs, path, 'object')
    
    Wavfile.data = norm(Wavfile.data)
    TmpWav = WavData(
        Wavfile.fs, Wavfile.data[0:10025], Wavfile.className)
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
    # print ('noisefactor =', noiseFactor)
    x = sum(Sxx)
    # print ('szum =', noise)
    x = rms(x, 5)  # opcja
    for i in range(0, len(x)):
        if x[i] <= noiseFactor * noise: # ile razy wieksze od szumu 
            x[i] = 0
        else:
            x[i] = 1
    # x = z
    # plotWav(x)
    plt.plot(x, 'b')

    s = []  # tablica poczatkow i koncow zdarzen
    cff = int(float(len(Wavfile.data)) / float(len(x)))
    if x[0] > 0: # na wypadek jakby plik zaczal sie od dzwieku
        s.append(0)
    for i in range(1, len(x)):
        if x[i] != x[i-1]:
            s.append(i * cff)

    for i in range(len(s)):
        offset = 2000
        if i % 2 == 0:
            s[i] += offset
        else:
            s[i] -= offset

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
        if mode == 'path':
            eTime = time.asctime(time.localtime(st[ST_MTIME] + duration[0])) # dla operacji na pliku
        elif mode == 'object':
            eTime = duration
        eTmp = AcousticEvent(Tmp, duration, eTime)
        eventDict[name] = eTmp

    return eventDict

def eventDetectionSlidingWindow(data, mode='object', fs=44100):
    q = queue.Queue() 
    eventDict = {}
    if mode == 'path':
        Wavfile = readData(data) # czyta z pliku 
        st = os.stat(data) # dla pliku
    elif mode == 'object':
        Wavfile = WavData(fs, data, 'object')
    
    f, t, Sxx = specCalculate(Wavfile.data, Wavfile.fs)
    f1, t1, Sxx1 = specCalculate(Wavfile.data[0:1050], Wavfile.fs)
    Tested = Spectro(f, t, Sxx)
    A = Spectro(f1, t1, Sxx1)
    x = slidingWindow(Tested,A)
    
    z = np.array([])

    for i in range(len(x)-1):
        z = np.append(z, x[i + 1] - x[i])
    
    z = norm(z)
    z = abs(z)
    z = rms(z, 10)
    
    for i in range(len(z)):    
        if (z[i]) > 0.02:
            z[i] = 1
        else:
            z[i] = 0

    plt.plot(z, 'r')
    
    s = []  # tablica poczatkow i koncow zdarzen
    cff = int(float(len(Wavfile.data)) / float(len(z)))
    if z[0] > 0: # na wypadek jakby plik zaczal sie od dzwieku
        s.append(0)
    for i in range(1, len(z)):
        if z[i] != z[i-1]:
            s.append(i * cff)

    # for i in range(len(s)): # rozszerzenie przedzialow 
    #     offset = 2000
    #     if i % 2 == 0:
    #         s[i] += offset
    #     else:
    #         s[i] -= offset

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
        if mode == 'path':
            eTime = time.asctime(time.localtime(st[ST_MTIME] + duration[0])) # dla operacji na pliku
        elif mode == 'object':
            eTime = duration
        eTmp = AcousticEvent(Tmp, duration, eTime)
        eventDict[name] = eTmp
        q.put(eTmp)


    return eventDict, q

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
            distance = min(slidingWindow(
                spectroDict[spec], EventDict[event].Spec)) # TODO zabezpieczyc jak nie ma plikow .spec
            if distance < minDistance:
                minDistance = distance
                EventDict[event].className = spec[0:-5]
        print (distance, ":\t", EventDict[event].printInfo())

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


def audioRecord(fs=44100, duration=5):
    '''Nagrywa dzwiek i zwradca WavData'''
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    record = WavData(fs, myrecording[:,0], '')
    return record


# --------------------------------- MAIN :) --------------------------------


# a = readAllSpecFiles()
# d = createAllSampleSpectogram('add')
# pngfromSpec()
# X = audioRecord()
# writeWavFile(X, 'nagranie.wav')
# dir = "nagranie.wav"
# wav = readData(dir)

# di, q = eventDetectionSlidingWindow(dir, mode='path')
# d = eventDetection(dir, mode='path')
# writeAllEvents(d)
# writeAllEvents(di)

# # d = classyfication(d)
# # writeAllEvents(d)

# # wav = readData(dir)
# # # wav.data = norm(wav.data)
# # fs =44100
# # duration = 5.5  # seconds
# # # myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)

# # # sd.wait()
# # # d = eventDetection(myrecording[:,0])
# # # sd.play(myrecording, fs)
# # # sd.wait()
# # # plotWav(myrecording, 'nagranie')
# plotWav(wav.data)



q = queue.Queue()

def putStreaIntoQueue():
    def callback(indata, frames, time, status):
               
        if status:
            text = ' ' + str(status) + ' '
            print(text)
        if any(indata):
            print(len(indata))
            q.put(indata.copy())
        else:
            print('no input')

    with sd.InputStream(channels=1, callback=callback,
                        blocksize=int(44100 / 4),
                        samplerate=44100):
        while True:
            response = input()
            print(response)
            if response in ('', 'q', 'Q'):
                break


def realTimeEventDetection():

    x = np.array([])
    print('foo1')
    while True:
        # time.sleep(0.1)
        if q.qsize() > 0:
            A = q.get()
            x = np.append(x, A)
            
            if len(x) >= 44100 * 3:
                wav = WavData(44100, x, 'plik')
                plt.plot(x)
                
                x = []
                detected, k = eventDetectionSlidingWindow(wav.data)
                # detected = classyfication(detected)
                writeAllEvents(detected)
                print('zapisałem', len(detected),  'plikow')
                plt.show()
        pass
    # d = eventDetectionSlidingWindow(x)
    # plotWav(x)
    # plt.show()        

# def foo3():
#     x = np.array([])
#     while True:
#         # time.sleep(0.1)
#         if q.qsize() > 3:
#             time.sleep(0.1)
#             A = q.get()
#             # print (A)
#             x = np.append(x, A[:,0])
            
#             if len(x) >= 44100 * 5:
#                 # plt.plot(x)
#                 # plt.show()
#                 wav = WavData(44100, x, 'plik')
#                 x = []
#                 writeWavFile(wav, 'plik.wav')
#                 print('zapisałem plik')
#     pass

thread1 = Thread( target=putStreaIntoQueue)
thread2 = Thread( target=realTimeEventDetection)
# thread3 = Thread( target=foo3)

thread1.start()
thread2.start()
# thread3.start()
thread1.join()
thread2.join()

print('done')

plt.show()


