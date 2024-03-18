import math
import os
import random
import av
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import constants as cst
from utils.utils import save_audio

def get_duration_sec(file, cache=False):
    try:
        with open(file + ".dur", "r") as f:
            duration = float(f.readline().strip("\n"))
        return duration
    except:
        container = av.open(file)
        audio = container.streams.get(audio=0)[0]
        duration = audio.duration * float(audio.time_base)
        if cache:
            with open(file + ".dur", "w") as f:
                f.write(str(duration) + "\n")
        return duration


#class ported from https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/main/data.py
class MultiSourceDataset(Dataset):
    def __init__(self, sr, channels, min_duration, max_duration, aug_shift, sample_length, audio_files_dir, stems, z_score):
        super().__init__()
        self.sr = sr
        self.channels = channels
        self.min_duration = min_duration or math.ceil(sample_length / sr)
        self.max_duration = max_duration or math.inf
        self.sample_length = sample_length
        self.audio_files_dir = audio_files_dir
        self.stems = stems
        self.z_score = z_score
        assert (
                sample_length / sr < self.min_duration
        ), f"Sample length {sample_length} per sr {sr} ({sample_length / sr:.2f}) should be shorter than min duration {self.min_duration}"
        self.aug_shift = aug_shift
        self.init_dataset()

    def init_dataset(self):
        # Load list of tracks and starts/durations
        tracks = os.listdir(self.audio_files_dir)
        #print(f"Found {len(tracks)} tracks.")
        self.filter(tracks)


    def __getitem__(self, item):
        index, offset = self.get_index_offset(item)
        wav = self.get_song_chunk(index, offset)
        # print(f"item: {item}, track: {index}, track: {self.tracks[index]},  offset: {offset},  cumsum: {self.cumsum[index]}, duration: {self.durations[index]}")
        if self.z_score:
            return (torch.from_numpy(wav) - cst.MEAN) / cst.STD
        else:
            return torch.from_numpy(wav)
        
    
    def get_index_offset(self, item):
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length // 2
        shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0
        offset = item * self.sample_length + shift  # Note we centred shifts, so adding now
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], f"Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}"

        index = np.searchsorted(self.cumsum, midpoint)  # index <-> midpoint of interval lies in this song
        start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[index]  # start and end of current song
        assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"

        if offset > end - self.sample_length:  # Going over song
            offset = max(start, offset - half_interval, end - self.sample_length) #TODO to ask why is was without end - sample.length
        elif offset < start:  # Going under song
            offset = min(end - self.sample_length, offset + half_interval, start)
        assert (
                start <= offset <= end - self.sample_length
        ), f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"

        offset = offset - start
        return index, offset

    def get_song_chunk(self, index, offset):
        track_name, total_length = self.tracks[index], self.durations[index]
        data_list = []
        for stem in self.stems:
            path_tmp = os.path.join(self.audio_files_dir, track_name, f'{stem}.wav')
            data, sr = self._load_audio(path_tmp, sr=self.sr, offset=offset, duration=self.sample_length, approx=True)
            # because the target channel is set to one and this is a stereo sound with two channel we do a mean across
            # the two channels to get the final output Channel
            #save_audio(data, f"{track_name}_{stem}_original")
            data = 0.5 * data[0:1, :] + 0.5 * data[1:, :]
            #save_audio(data, f"{track_name}_{stem}_onechannel")
            #save_audio((data-cst.MEAN) / cst.STD, f"{track_name}_{stem}_zscore")
            assert data.shape == (
                self.channels,
                self.sample_length,
            ), f"Expected {(self.channels, self.sample_length)}, got {data.shape}"
            data_list.append(data)

        return np.concatenate(data_list, axis=0)


    def _load_audio(self, file, offset, duration, sr=None, resample=False, approx=False, time_base='samples',
                   check_duration=True):
        if time_base == 'sec':
            offset = offset * sr
            duration = duration * sr
        # Loads at target sr, stereo channels, seeks from offset, and stops after duration
        container = av.open(file)
        audio = container.streams.get(audio=0)[0]  # Only first audio stream
        audio_duration = audio.duration * float(audio.time_base)
        if approx:
            if offset + duration > audio_duration * sr:
                # Move back one window. Cap at audio_duration
                offset = min(audio_duration * sr - duration, offset - duration)
        else:
            if check_duration:
                assert offset + duration <= audio_duration * sr, f'End {offset + duration} beyond duration {audio_duration * sr}'
        if resample:
            resampler = av.AudioResampler(format='fltp', layout='stereo', rate=sr)
        else:
            assert sr == audio.sample_rate
        offset = int(offset / sr / float(
            audio.time_base))  # int(offset / float(audio.time_base)) # Use units of time_base for seeking
        sig = np.zeros((2, duration), dtype=np.float32)
        container.seek(offset, stream=audio)
        total_read = 0
        for frame in container.decode(audio=0):  # Only first audio stream
            if resample:
                frame.pts = None
                frame = resampler.resample(frame)
            frame = frame.to_ndarray(format='fltp')  # Convert to floats and not int16
            read = frame.shape[-1]
            if total_read + read > duration:
                read = duration - total_read
            sig[:, total_read:total_read + read] = frame[:, :read]
            total_read += read
            if total_read == duration:
                break
        assert total_read <= duration, f'Expected {duration} frames, got {total_read}'
        return sig, sr

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def filter(self, tracks):
        # Remove files too short or too long
        keep = []
        durations = []
        for track in tracks:

            track_dir = os.path.join(self.audio_files_dir, track)
            files = librosa.util.find_files(f"{track_dir}", ext=["mp3", "opus", "m4a", "aac", "wav"])

            # skip if there are not all the sources
            if len(files) != 4:
                continue

            durations_track = np.array(
                [get_duration_sec(file, cache=True) * self.sr for file in files])  # Could be approximate

            # skip if there is a source that is shorter than minimum track length
            if (durations_track / self.sr < self.min_duration).any():
                continue

            # skip if there is a source that is longer than maximum track length
            if (durations_track / self.sr >= self.max_duration).any():
                continue

            # skip if in the track the different sources have different lengths
            if not (durations_track == durations_track[0]).all():
                #print(f"{track} skipped because sources are not aligned!")
                #print(durations_track)
                continue
            keep.append(track)
            durations.append(durations_track[0])

        #print(f"self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}")
        #print(f"Keeping {len(keep)} of {len(tracks)} tracks")
        self.tracks = keep
        self.durations = np.array(durations)
        self.cumsum = np.cumsum(self.durations)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_train = train_dataset
        self.data_val = val_dataset
        self.data_test = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.g = torch.Generator()
        self.g.manual_seed(cst.SEED)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=self.g
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size*2,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=self.g
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=self.g
        )
    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)