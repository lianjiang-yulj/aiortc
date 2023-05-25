import argparse
import asyncio
import datetime
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame, AudioFrame
import time
import queue
import numpy as np
import fractions
import nest_asyncio
from typing import Tuple, Union
from aiortc.mediastreams import AUDIO_PTIME, VIDEO_CLOCK_RATE, VIDEO_PTIME, VIDEO_TIME_BASE, MediaStreamError

# AUDIO_PTIME = 0.020  # 20ms audio packetization
# VIDEO_CLOCK_RATE = 90000
# VIDEO_PTIME = 1 / 30  # 30fps
# VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)

AUDIO_SAMPLERATE = 48000 # 8000
AUDIO_TIME_BASE = fractions.Fraction(1, AUDIO_SAMPLERATE)
VIDEO_PTIME = 1 / 30  # 30fps
print(f'mediastreams variables:  AUDIO_PTIME({AUDIO_PTIME}), VIDEO_CLOCK_RATE({VIDEO_CLOCK_RATE}), VIDEO_PTIME({VIDEO_PTIME}), VIDEO_TIME_BASE({VIDEO_TIME_BASE}), AUDIO_SAMPLERATE({AUDIO_SAMPLERATE}), AUDIO_TIME_BASE({AUDIO_TIME_BASE})')


import sys

import cv2

import multiprocessing as mp

import signal
import queue
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

user_python_path=[
        'yulj/PycharmProjects/pythonProject',
        'yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN',
        'yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/Wav2Lip-master'
        ]

home_path = os.environ['HOME']
for p in user_python_path:
    sys.path.append(os.path.join(home_path, p))

#print("os.environ['PYTHONPATH']={0}".format(os.environ['PYTHONPATH']))
print("->>>>>>>>>>>>>>>>os.environ['PATH']={0}".format(os.environ['PATH']))
print("->>>>>>>>>>>>>>>>sys.path={0}".format(sys.path))
import inference_rt
# from hparams import hparams as hp
# import audio
# import librosa

print(f'import ok')

nest_asyncio.apply()

ROOT = os.path.dirname(__file__)
#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='## %Y-%m-%d %H:%M:%S')

srv_logger = logging.getLogger("pc")
srv_logger.info('import success')
pcs = set()
relay = MediaRelay()

global relay_audio
relay_audio = MediaRelay()
print(f'global relay_audio', relay_audio)

#存放音频
global spkrSampleQueue
spkrSampleQueue = queue.Queue(0)
print(f'global spkrSampleQueue', spkrSampleQueue)

#存放视频
global gvideoSampleQueue
gvideoSampleQueue = queue.Queue(0)
print(f'global gvideoSampleQueue', gvideoSampleQueue)

global blackHole
blackHole = MediaBlackhole()
print(f'global blackHole', blackHole)

argv = [
            '--checkpoint_path',
            os.path.join(home_path,'yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/Wav2Lip-master/checkpoints/wav2lip_gan.pth'),
            '--face',
            os.path.join(home_path, 'yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/inputs/7a3fd733fb4e90aac3ad39d6e4ed0251.mp4'),
            '--audio',
            os.path.join(home_path, 'yulj/PycharmProjects/pythonProject/source/aiortc/examples/server/5a7bdba7-563a-44ce-95fa-ded35217310b1_2.wav'),
            '--outfile',
            os.path.join(home_path, 'yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/Wav2Lip-master/results/trim_ipseek_copy_result_3.mp4'),
            '--fps',
            '25',
            '--face_det_batch_size',
            '2',
            '--wav2lip_batch_size', # 一次处理8帧
            '8'
        ]

global rt_wav2lip
rt_wav2lip = inference_rt.RtWav2Lip(argv, spkrSampleQueue, gvideoSampleQueue)
rt_wav2lip.main()

def put_frame():
    global rt_wav2lip

    while True:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        rt_wav2lip.inference()
        new_loop.close()
        if (rt_wav2lip.get_stop()):
            return

        time.sleep(0.1)

global main_thread
main_thread = threading.Thread(target=put_frame, group=None,
                               args=(), daemon=True)
main_thread.start()
global main_thread_id
main_thread_id = main_thread.ident

def closeRtWav():
    rt_wav2lip.set_stop(True)
    signal.pthread_kill(main_thread.ident, signal.SIGKILL)
    main_thread.terminate()
    main_thread.join()

def handle_user_signal(sig, frame_stack):

    print(f'->>>>>>>>>>>>>>>>main pid {os.getpid()}, sub thread:{main_thread_id} recieve kill signal: ', sig, frame_stack)


    if int(sig) == int(signal.SIGUSR1):
        p = not rt_wav2lip.get_pause()
        print(f'recieve user signal SIGUSR1, set pause = {p}')
        rt_wav2lip.set_pause(p)
        return
    elif int(sig) == int(signal.SIGUSR2):
        print(f'recieve user signal SIGUSR2, set stop = True')
        rt_wav2lip.set_stop(True)


    #global stop_threads #使用全局的 stop_threads
    stop_threads = True
    closeRtWav()

    return

    children = mp.active_children()

    print(f'->>>>>>>>>>>>>>>>child pids: ', children)
    for child in children:
        print(f'->>>>>>>>>>>>>>>>kill wav process pid: {child.pid}')
        child.terminate()
        #child.kill()
        os.kill(child.pid, signal.SIGKILL)
        child.join()
        os.waitpid(child.pid, 0)
        g_mp.terminate()
        g_mp.kill()
        g_mp.join()
        #
        #f

    # if g_mp is not None:
    #     try:
    #         print(f'->>>>>>>>>>>>>>>> kill mp: {g_mp.pid}, {g_wav_pid}')
    #
    #         g_mp.terminate()
    #         g_mp.kill()
    #     except Exception as e:
    #         print(f'->>>>>>>>>>>>>>>>未知错误 %s' %(e))
    #         os.kill(g_wav_pid, signal.SIGKILL)


    #   exit
    sys.exit(0)


#signal.signal(signal.SIGINT, handle_user_signal)
#signal.signal(signal.SIGTERM, handle_user_signal)
signal.signal(signal.SIGUSR1, handle_user_signal)
signal.signal(signal.SIGUSR2, handle_user_signal)

print(f'->>>>>>>>>>>>>>>>global main pid in front {os.getpid()}')


class MicStreamTrack(MediaStreamTrack):
    """
    An audio stream object for the mic audio from the client
    """
    kind = "audio"

    def __init__(self, track, player):
        super().__init__()
        self.track = track
        self.player = player

        srv_logger.info("MicStreamTrack initialized")
        self.call_count = 0
        self.print_count = 250
        self.start_time = 0
        self.end_time = 0
        # self.sr = hp.sample_rate  # 16000
        # print("hparams sample_rate:", self.sr)


        path = "/home/ps/yulj/PycharmProjects/pythonProject/source/aiortc/examples/server/5a7bdba7-563a-44ce-95fa-ded35217310b1_2.wav"
        # self.wav = audio.load_wav(path, self.sr) #librosa.core.load(path, sr=self.sr, dtype=np.int16)[0]#
        # self.hop_length = int(0.02 * self.sr)
        # self.frame_length = 1920 #4 * self.hop_length
        # self.audio_frames = []
        # self.gen_frames(self.wav, self.frame_length, self.hop_length)
        # self.audio_frame_index = 0

        self.myplayer = MediaPlayer(file=path, loop=False)

    # def gen_frames(self, waveform, frame_length=2048, hop_length=512):
    #     self.audio_frames = librosa.util.frame(waveform, frame_length=frame_length, hop_length=hop_length).T
    #     print(f"音频帧的frames params: frame_length={frame_length}, hop_length={hop_length}", )
    #     print(f"音频帧的frames：", self.audio_frames)
    #     # 打印音频帧的形状
    #     print(f"音频帧的形状：", self.audio_frames.shape)
    #     print(f"音频帧的frame 0：", self.audio_frames[0])
    #     print(f"音频帧的frame 0 type：", type(self.audio_frames[0]))
    #     print(f"音频帧的frame 0 data type：", type(self.audio_frames[0][0]))
    #     print(f"音频帧的0形状：", self.audio_frames[0].shape)
    #     print("---------------------------------")

    async def recv(self):
        global spkrSampleQueue

        if self.start_time == 0:
            self.start_time = int(time.time())
            self.a_start_time = self.start_time
            print(
                f'MicStreamTrack myplayer.audio.recv  start time:{self.start_time}')

        # Get a new PyAV frame，
        # frame = await self.track.recv()
        frame = await my_track_consume(self.track)
        #arr = frame.to_ndarray(format='s16', layout='stereo')
        #print(f'测试测试 track frame：', arr, np.shape(arr))
        frame = await my_track_consume(self.player.audio)
        #arr = frame.to_ndarray(format='s16', layout='stereo')

        #print(f'测试测试 player frame：',arr, np.shape(arr))

        #frame = self.audio_frames[self.audio_frame_index] if self.audio_frame_index < len(self.audio_frames) else None

        # 将波形数据转换为 int16 类型
        #frame = (frame * np.iinfo(np.int16).max).astype(np.int16)
        #frame = frame.reshape(1, len(frame))
        #print(f'测试测试 wav frame：',frame, np.shape(arr))

        #self.audio_frame_index += 1

        frame = await my_track_consume(self.myplayer.audio)
        return

        if frame is None and self.call_count % self.print_count == 0 and self.call_count > 0:
            # 麦克风永远有数据
            print(f'!!!!!!!!!!!!!!! ERROR: MicStreamTrack track.audio.recv  [{self.call_count}]  none frame!!!!!!!!!!!!!!!')

        # # Convert to float32 numpy array
        floatArray = None
        if frame is not None: floatArray = frame.to_ndarray(format='s16', layout='stereo')


        # Put these samples into the mic queue
        spkrSampleQueue.put(floatArray)
        
        if (self.call_count % self.print_count == 0 and self.call_count > 0):
            self.end_time = int(time.time())
            print(f'MicStreamTrack track.audio.recv [{self.call_count}] frame, data: {floatArray}, data_shape: {np.shape(floatArray)}, fps: {self.print_count / (self.end_time - self.start_time) }, frame: {frame}, queue({spkrSampleQueue}) size: {spkrSampleQueue.qsize()}, readyState: {self.readyState}')
            self.start_time = self.end_time

        self.call_count += 1

        return


async def my_track_consume(track):
    if track is None:
        return None

    frame = None
    try:
        frame = await track.recv()
    except MediaStreamError as mse:
        #print(f'!!!!!!!!!!!!!ERROR: recieve MediaStreamError, {track}', mse)
        return None
    except Exception as e:
        #print(f'!!!!!!!!!!!!!ERROR: recieve Exception, {track}', e)
        return None
    return frame


async def putFrameToQueue(fromClass, call_count, print_count, batch = False):
    global player
    # frame = await player.audio.recv()
    frame = await my_track_consume(player.audio)
    # player 如果播放 完成 会阻塞

    # if frame is None and call_count % print_count == 0 and call_count > 0:
    if frame is None:
        if not batch or (batch and call_count % print_count == 0 and call_count > 0):
            print(f'!!!!!!!!!!!!!!! ERROR: {fromClass} player.audio.recv [{call_count}] none frame!!!!!!!!!!!!!!!')

    # frame 默认format='s16', layout='stereo'， to_ndarray参数不s
    floatArray = None
    if frame is not None:
        floatArray = frame.to_ndarray(format='s16', layout='stereo')

    global spkrSampleQueue
    spkrSampleQueue.put(floatArray)

    return frame, floatArray


class SpkrStreamTrack(MediaStreamTrack):
    """
    An audio stream object for the speaker data from the server
    """
    kind = "audio"

    _start: float
    _timestamp: int

    def __init__(self):
        super().__init__()
        self.samplerate = AUDIO_SAMPLERATE
        self.samples = int(AUDIO_PTIME * self.samplerate) #960
        self.audio_time_base = AUDIO_TIME_BASE
        self.print_count = 250
        srv_logger.info("SpkrStreamTrack initialized")
        self.call_count = 0
        self.start_time = 0
        self.end_time = 0

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:

        if self.readyState != "live":
            raise MediaStreamError

        # Handle timestamps properly
        if hasattr(self, "_timestamp"):
            self._timestamp += self.samples
            wait = self._start + (self._timestamp / self.samplerate) - time.time()
            if (wait >= 0.00001):
                #print(f'SpkrStreamTrack await {wait} seconds')
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        return self._timestamp, AUDIO_TIME_BASE

    async def recv(self):
        global spkrSampleQueue

        if self.start_time == 0:
            self.start_time = int(time.time())


        # create empty data by default
        data = np.zeros(self.samples).astype(np.int16)
        data = data.reshape(data.shape[0], -1).T

        # test put
        #putFrameToQueue("SpkrStreamTrack", self.call_count, self.print_count, batch = False)

        is_data_has = False
        # Only get speaker data if we have some in the buffer
        if spkrSampleQueue.qsize() > 0:
            try:
                #srv_logger.info("Getting speaker samples from queue")
                # data = np.zeros(self.samples).astype(np.int16)
                # data = data.reshape(data.shape[0], -1).T

                qdata = spkrSampleQueue.get_nowait()
                if qdata is not None:
                    data = qdata
                    is_data_has = True
            except queue.Empty:
                #srv_logger.info("Getting empty speaker samples from queue")
                # To convert to a mono audio frame, we need the array to be an array of single-value arrays for each sample (annoying)
                #data = data.reshape(data.shape[0], -1).T
                pass
        else:
            #srv_logger.info("Getting empty speaker samples from queue")
            # To convert to a mono audio frame, we need the array to be an array of single-value arrays for each sample (annoying)
            #data = data.reshape(data.shape[0], -1).T
            pass


        # Create audio frame
        layout = 'mono'
        if is_data_has:
            layout = 'stereo'

        frame = AudioFrame.from_ndarray(data, format='s16', layout=layout)

        pts, time_base = await self.next_timestamp()
        # Update time stuff
        frame.pts = pts
        frame.sample_rate = self.samplerate
        frame.time_base = time_base

        if (self.call_count % self.print_count == 0 and self.call_count > 0):
            self.end_time = int(time.time())
            print(
                f'SpkrStreamTrack player.audio.recv [{self.call_count}] frame({frame}) shape: {np.shape(data)}, is_data_has: {is_data_has}, data: {data} from mic queue({spkrSampleQueue}) qsize {spkrSampleQueue.qsize()}, video queue({gvideoSampleQueue}) qsize {gvideoSampleQueue.qsize()}, fps: {self.print_count / (self.end_time - self.start_time)}, readyState: {self.readyState}')
            self.start_time = self.end_time

        self.call_count += 1

            # Return
        return frame


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            if (wait >= 0.00001):
                #print(f'VideoTransformTrack await {wait} seconds')
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.print_count = 150
        self.call_count = 0
        self.start_time = 0
        self.end_time = 0
        self.last_frame = None
        srv_logger.info("VideoTransformTrack initialized, transform:{}".format(transform))

    async def recv(self):
        global gvideoSampleQueue
        if self.start_time == 0:
            self.start_time = int(time.time())
            print("global gvideoSampleQueue: {gvideoSampleQueue}")

        # frame = await self.track.recv()
        # frame = await my_track_consume(self.track)
        frame = None
        try:
            # srv_logger.info("Getting speaker samples from queue")
            # data = np.zeros(self.samples).astype(np.int16)
            # data = data.reshape(data.shape[0], -1).T


            frame_array = gvideoSampleQueue.get_nowait()

            if frame_array is  None:
                frame  = self.last_frame
            else:
                frame = VideoFrame.from_ndarray(frame_array, format="bgr24")
                self.last_frame = frame

        except queue.Empty:
            # srv_logger.info("Getting empty speaker samples from queue")
            # To convert to a mono audio frame, we need the array to be an array of single-value arrays for each sample (annoying)
            # data = data.reshape(data.shape[0], -1).T
            frame = self.last_frame
            pass
        except Exception as e:
            print(f'->>>>>>>>>>>>>>>>VideoFrame %s' %(e))
            print(f'->>>>>>>>>>>>>>>>global gvideoSampleQueue: {gvideoSampleQueue}')
            print(f'->>>>>>>>>>>>>>>>VideoFrame %s, global gvideoSampleQueue: {gvideoSampleQueue}' %(e))

        if frame is None and self.call_count % self.print_count == 0 and self.call_count > 0:
            print(f'!!!!!!!!!!!!!!! ERROR: VideoTransformTrack recv [{self.call_count}] none frame!!!!!!!!!!!!!!!')

        new_frame = None
        if self.transform == "cartoon" and frame is not None:
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

        elif self.transform == "edges"  and frame is not None:
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

        elif self.transform == "rotate"  and frame is not None:
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

        else:
            new_frame = frame
            if new_frame is None:
                new_frame = VideoFrame(width=640, height=480)
                for p in new_frame.planes:
                    p.update(bytes(p.buffer_size))

        pts, time_base = await self.next_timestamp()

        # frame = VideoFrame(width=640, height=480)
        # for p in frame.planes:
        #     p.update(bytes(p.buffer_size))
        new_frame.pts = pts
        new_frame.time_base = time_base


        if (self.call_count % self.print_count == 0 and self.call_count > 0):
            self.end_time = int(time.time())
            print(
                f'VideoTransformTrack recv [{self.call_count}] frame({new_frame})  from video queue({gvideoSampleQueue}) qsize {gvideoSampleQueue.qsize()}, mic queue({spkrSampleQueue}) qsize {spkrSampleQueue.qsize()}, fps: {self.print_count / (self.end_time - self.start_time)}, readyState: {self.readyState}')
            self.start_time = self.end_time

        self.call_count += 1

        return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def start_blackhole(black_hole):
    await black_hole.start()

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        srv_logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    global player
    player = MediaPlayer(file = os.path.join(ROOT, "5a7bdba7-563a-44ce-95fa-ded35217310b1_2.wav"), loop=True)
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            # pc.addTrack(player.audio)
            # 一定要relay，这样可以重复消费
            recorder.addTrack(relay_audio.subscribe(track))


            # Create the mic stream for this track, and add its output to a media blackhole so it starts
            micTrack = MicStreamTrack(relay_audio.subscribe(track), player)
            blackHole.addTrack(micTrack)
            log_info("Added mic blackHole track")

            start_begin = int(time.time())
            log_info("Starting mic blackHole tracking begin at %s", start_begin)

            #blackHole.start()

            loop = asyncio.get_event_loop()
            # blackHole 一直会while 消费 track
            #result = loop.run_until_complete(start_blackhole(blackHole))
            asyncio.run(start_blackhole(blackHole))
            #loop.close()

            start_end = int(time.time())
            log_info("Started mic blackHole track end at %s, cost: %s", start_end, start_end - start_begin)

            # Create the speaker track
            spkrTrack = SpkrStreamTrack()
            pc.addTrack(spkrTrack)
            log_info("Added speaker track")

        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            log_info("Added video track")

            # if args.record_to:
            #     recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()
            log_info("Track %s ended, recorder.stopped, see record file: %s", track.kind, args.record_to)





    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    print(f'->>>>>>>>>>>>>>>>global main pid in rtc {os.getpid()}')
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )




async def on_shutdown(app):
    global blackHole
    await blackHole.stop()
    srv_logger.info("global on_shutdown: blackHole")

    # global player
    # player.audio.stop()
    # srv_logger.info("global on_shutdown: player")


    closeRtWav()
    srv_logger.info("global on_shutdown: closeRtWav")

    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

    srv_logger.info("global on_shutdown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
